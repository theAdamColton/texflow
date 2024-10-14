import bpy

from texflow.client.depth import render_depth_map
from texflow.client.utils import select_obj
from texflow.tests.utils import TestCase, save_image


class TestClient(TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)

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
