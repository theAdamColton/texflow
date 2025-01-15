import numpy as np
import math
import bpy
import bmesh


from ..client.uv import uv_proj
from ..client.depth import render_depth_map
from ..client.utils import select_obj
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

        depth_map, occupancy = render_depth_map(obj, camera)

        test_dir = self.get_test_dir()
        save_image(depth_map, test_dir / "depth_map.png")

        height, width = depth_map.shape

        # the middle pixel should not be max distance
        self.assertLess(depth_map[height // 2, width // 2].item(), 1.0)

    def test_render_depth_map_w_extra_distance(self):
        bpy.ops.mesh.primitive_ico_sphere_add()
        obj = bpy.context.object
        select_obj(obj)
        bpy.ops.object.camera_add(location=(0.0, -3.0, 0.0), rotation=(1.5, 0, 0))
        camera = bpy.context.active_object
        extra_background_distance = 0.1
        depth_map, occupancy = render_depth_map(
            obj,
            camera,
            extra_background_distance=extra_background_distance,
        )

        self.assertAlmostEqual(
            1 - extra_background_distance, depth_map[occupancy].max().item(), 1
        )

    def test_uv_proj(self):
        """
        tests that all the u,vs projected by this cube are visible from the camera
        """
        bpy.ops.object.camera_add(
            location=(0.0, -9.0, 0.0), rotation=(math.pi / 2, 0, 0)
        )
        camera_obj = bpy.context.active_object

        bpy.ops.mesh.primitive_cube_add()
        obj = bpy.context.active_object
        select_obj(obj)

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        uv_proj(obj, camera_obj)

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

        extra_background_distance = 0.0
        depth_map, occupancy = render_depth_map(
            obj,
            camera,
            extra_background_distance=extra_background_distance,
        )

        im = to_image16(depth_map)
        rec_depth_map = np.asarray(im) / (2**16 - 1)
        self.assertTrue(np.allclose(depth_map, rec_depth_map, 1e-4, 1e-4))

    def test_render_depth_and_geo_image_suzanne(self):
        bpy.ops.mesh.primitive_monkey_add()
        obj = bpy.context.object
        select_obj(obj)
        bpy.ops.object.camera_add(location=(6.0, -6.0, 4.0), rotation=(1.1, 0, 0.8))
        camera = bpy.context.active_object
        extra_background_distance = 0.0
        depth_map, occupancy = render_depth_map(
            obj,
            camera,
            extra_background_distance=extra_background_distance,
        )

        im = to_image16(depth_map)
        im.save(self.get_test_dir() / "depth_image.png")
        height, width = depth_map.shape

        select_obj(obj)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        uv_layer = uv_proj(obj, camera)

        me = obj.data
        select_obj(obj)
        bm = bmesh.from_edit_mesh(me)

        uv_layer = bm.loops.layers.uv.verify()

        # renders the uv coords as dots on an image
        uv_image = np.zeros((height, width))
        for face in bm.faces:
            for loop in face.loops:
                loop_uv = loop[uv_layer]
                uv = loop_uv.uv
                x, y = uv.x, uv.y
                # need to flip height coord
                y = 1 - y
                # x and y refer to width and height respectively
                i, j = int(y * height), int(x * width)
                uv_image[i, j] = 1
        uv_image = to_image16(uv_image)
        uv_image.save(self.get_test_dir() / "uv_image.png")
