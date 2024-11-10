import nest_asyncio
import logging
import aiohttp
import uuid
from aiohttp.test_utils import AioHTTPTestCase
import asyncio
from aiohttp import web
import numpy as np
import math
import bpy
import bmesh

from texflow.state import TexflowStatus

from ..client.uv import uv_proj
from ..client.depth import render_depth_map
from ..client.utils import select_obj
from ..client.ui import TexflowAsyncOperator, get_texflow_state
from ..client import register, unregister
from ..tests.utils import TestCase, save_image
from ..client.utils import to_image16


def _setUpTexflow():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        unregister()
    except:
        pass
    register()


def _tearDownTexflow():
    unregister()
    bpy.ops.wm.read_factory_settings(use_empty=True)


class TestClient(TestCase):
    def setUp(self):
        _setUpTexflow()

    def tearDown(self):
        _tearDownTexflow()

    def test_render_depth_map(self):
        bpy.ops.mesh.primitive_ico_sphere_add()
        obj = bpy.context.object
        select_obj(obj)
        bpy.ops.object.camera_add(location=(0.0, -3.0, 0.0), rotation=(1.5, 0, 0))
        camera = bpy.context.active_object

        height, width = 256, 256
        depth_map, occupancy = render_depth_map(obj, camera, height=height, width=width)

        test_dir = self.get_test_dir()
        save_image(depth_map, test_dir / "depth_map.png")

        self.assertEqual((height, width), depth_map.shape)

        # the middle pixel should not be max distance
        self.assertLess(depth_map[height // 2, width // 2].item(), 1.0)

    def test_render_depth_map_w_extra_distance(self):
        bpy.ops.mesh.primitive_ico_sphere_add()
        obj = bpy.context.object
        select_obj(obj)
        bpy.ops.object.camera_add(location=(0.0, -3.0, 0.0), rotation=(1.5, 0, 0))
        camera = bpy.context.active_object
        height, width = 256, 256

        extra_background_distance = 0.1
        depth_map, occupancy = render_depth_map(
            obj,
            camera,
            height=height,
            width=width,
            extra_background_distance=extra_background_distance,
        )

        self.assertAlmostEqual(
            1 - extra_background_distance, depth_map[occupancy].max().item(), 1
        )

    def test_uv_proj(self):
        """
        tests that all the u,vs projected by this cube are visible from the camera
        """
        height = 512
        width = 512
        bpy.ops.object.camera_add(
            location=(0.0, -4.0, 0.0), rotation=(math.pi / 2, 0, 0)
        )
        camera_obj = bpy.context.active_object

        bpy.ops.mesh.primitive_cube_add()
        obj = bpy.context.active_object
        select_obj(obj)

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        uv_proj(obj, camera_obj, height=height, width=width)

        me = obj.data
        select_obj(obj)
        bm = bmesh.from_edit_mesh(me)

        uv_layer = bm.loops.layers.uv.verify()

        # adjust uv coordinates
        for face in bm.faces:
            for loop in face.loops:
                loop_uv = loop[uv_layer]
                uv = loop_uv.uv
                self.assertGreaterEqual(uv.x, 0)
                self.assertGreaterEqual(uv.y, 0)
                self.assertLessEqual(uv.x, 1)
                self.assertLessEqual(uv.y, 1)

    def test_save_tiff(self):
        arr = np.random.random((32, 32))
        im = to_image16(arr)
        rec_arr = np.asarray(im) / (2**16 - 1)
        self.assertTrue(np.allclose(arr, rec_arr, 1e-4, 1e-4))

        arr = np.ones((32, 32))
        im = to_image16(arr)
        rec_arr = np.asarray(im) / (2**16 - 1)
        self.assertTrue(np.allclose(arr, rec_arr, 1e-4, 1e-4))

    def test_save_load_depth_tiff(self):
        bpy.ops.mesh.primitive_ico_sphere_add()
        obj = bpy.context.object
        select_obj(obj)
        bpy.ops.object.camera_add(location=(0.0, -3.0, 0.0), rotation=(1.5, 0, 0))
        camera = bpy.context.active_object
        height, width = 256, 256

        extra_background_distance = 0.0
        depth_map, occupancy = render_depth_map(
            obj,
            camera,
            height=height,
            width=width,
            extra_background_distance=extra_background_distance,
        )

        im = to_image16(depth_map)
        rec_depth_map = np.asarray(im) / (2**16 - 1)
        self.assertTrue(np.allclose(depth_map, rec_depth_map, 1e-4, 1e-4))


class TestClientServer(AioHTTPTestCase):
    depth_image_post = None
    sockets = dict()
    url = None

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        self.depth_image_post = None
        self.sockets = dict()
        self.url = f"127.0.0.1:{self.server.port}"
        _setUpTexflow()

    async def asyncTearDown(self) -> None:
        self.depth_image_post = None
        self.sockets = dict()
        self.url = None
        _tearDownTexflow()

        await super().asyncTearDown()

    async def send_json(self, event, data, sid=None):
        message = {"type": event, "data": data}

        if sid is None:
            sockets = list(self.sockets.values())
            for ws in sockets:
                await ws.send_json(message)
        elif sid in self.sockets:
            await self.sockets[sid].send_json(message)

    async def get_application(self):
        routes = web.RouteTableDef()

        @routes.post("/upload/image")
        async def upload_image(request):
            post = await request.post()
            self.depth_image_post = post
            return web.json_response(
                {
                    "name": "dummyfilename",
                    "subfolder": "some/subfolder",
                    "type": "some type",
                }
            )

        @routes.get("/ws")
        async def websocket_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            sid = request.rel_url.query.get("clientId", "")
            if sid:
                # Reusing existing session, remove old
                self.sockets.pop(sid, None)
            else:
                sid = uuid.uuid4().hex

            self.sockets[sid] = ws

            try:
                # Send initial state to the new client
                await self.send_json("status", {"status": dict(), "sid": sid}, sid)

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.ERROR:
                        logging.warning(
                            "ws connection closed with exception %s" % ws.exception()
                        )
            finally:
                self.sockets.pop(sid, None)
            return ws

        app = web.Application()
        app.add_routes(routes)
        return app

    async def test_connection_completes(self):
        texflow = bpy.context.scene.texflow
        texflow.comfyui_url = self.url
        bpy.ops.texflow.connect_to_comfy()

        async_loop_mgr = TexflowAsyncOperator.get_async_manager()
        self.assertIsNotNone(async_loop_mgr.loop)
        # needed for nested async loops
        nest_asyncio.apply(async_loop_mgr.loop)

        async with asyncio.timeout(2):
            while get_texflow_state().status != TexflowStatus.READY:
                await asyncio.sleep(1 / 100)
                async_loop_mgr.kick()

        self.assertEqual(get_texflow_state().status, TexflowStatus.READY)
        self.assertIsNotNone(get_texflow_state().client_id)

    async def test_render_depth_map_op(self):
        texflow = bpy.context.scene.texflow
        texflow.comfyui_url = self.url

        bpy.ops.mesh.primitive_ico_sphere_add()
        obj = bpy.context.object
        bpy.ops.object.camera_add(location=(0.0, -3.0, 0.0), rotation=(1.5, 0, 0))
        camera = bpy.context.active_object
        texflow.camera = camera
        select_obj(obj)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        texflow.height = 16
        texflow.width = 16

        bpy.ops.texflow.render_depth_image()

        async_loop_mgr = TexflowAsyncOperator.get_async_manager()
        self.assertIsNotNone(async_loop_mgr.loop)

        # needed for nested async loops
        nest_asyncio.apply(async_loop_mgr.loop)

        async with asyncio.timeout(2):
            while self.depth_image_post is None:
                await asyncio.sleep(1 / 100)
                async_loop_mgr.kick()

        self.assertIn("image", self.depth_image_post)

    async def test_connect_and_then_render(self):
        texflow = bpy.context.scene.texflow
        texflow.comfyui_url = self.url
        bpy.ops.texflow.connect_to_comfy()

        async_loop_mgr = TexflowAsyncOperator.get_async_manager()

        self.assertIsNotNone(async_loop_mgr.loop)
        # needed for nested async loops
        nest_asyncio.apply(async_loop_mgr.loop)

        async with asyncio.timeout(2):
            while get_texflow_state().status != TexflowStatus.READY:
                await asyncio.sleep(1 / 100)
                async_loop_mgr.kick()

        bpy.ops.mesh.primitive_ico_sphere_add()
        obj = bpy.context.object
        bpy.ops.object.camera_add(location=(0.0, -3.0, 0.0), rotation=(1.5, 0, 0))
        camera = bpy.context.active_object
        texflow.camera = camera
        select_obj(obj)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        texflow.height = 16
        texflow.width = 16

        bpy.ops.texflow.render_depth_image()
        async with asyncio.timeout(2):
            while self.depth_image_post is None:
                await asyncio.sleep(1 / 100)
                async_loop_mgr.kick()

        self.assertIn("image", self.depth_image_post)
