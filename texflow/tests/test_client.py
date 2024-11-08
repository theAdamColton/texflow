import uuid
from aiohttp.test_utils import AioHTTPTestCase
import asyncio
from aiohttp import web
import numpy as np
import io
import math
import time
import bpy
import bmesh

from ..client.uv import uv_proj
from ..client.depth import render_depth_map
from ..client.async_loop import kick_async_loop
from ..client.utils import select_obj
from ..client.ui import get_texflow_state
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
        rec_arr = np.asarray(im) / 2**16
        self.assertTrue(np.allclose(arr, rec_arr, 1e-4, 1e-4))


class TestClientServer(AioHTTPTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        _setUpTexflow()

    async def asyncTearDown(self) -> None:
        await super().asyncTearDown()
        _tearDownTexflow()

    async def get_application(self):
        routes = web.RouteTableDef()

        @routes.post("/upload/image")
        async def upload_image(request):
            post = await request.post()
            return web.json_response(
                {
                    "name": "dummyfilename",
                    "subfolder": "some/subfolder",
                    "type": "some type",
                }
            )

        app = web.Application()
        app.add_routes(routes)
        return app

    async def test_connect(self):
        pass

    async def test_render_depth_map(self):
        texflow = bpy.context.scene.texflow
        texflow.comfyui_url = str(self.client.make_url(""))
        bpy.ops.mesh.primitive_ico_sphere_add()
        obj = bpy.context.object
        bpy.ops.object.camera_add(location=(0.0, -3.0, 0.0), rotation=(1.5, 0, 0))
        camera = bpy.context.active_object
        texflow.camera = camera
        select_obj(obj)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.texflow.render_depth_image(height=16, width=16)

        # TODO hacky test sleep
        await asyncio.sleep(4)
