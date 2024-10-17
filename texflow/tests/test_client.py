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
from ..controller.pipe_utils import load_pipe


class TestClient(TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        try:
            unregister()
        except:
            pass
        register()

    def tearDown(self):
        unregister()
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

    def test_generate_interrupt(self):
        state = get_texflow_state()
        pipe = load_pipe(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            controlnet_models_or_paths=["hf-internal-testing/tiny-controlnet"],
        )
        state.pipe = pipe
        texflow_props = bpy.context.scene.texflow
        height = 64
        width = 64
        texflow_props.height = height
        texflow_props.width = width
        texflow_props.steps = 10
        bpy.ops.object.camera_add(location=(0.0, -3.0, 0.0), rotation=(1.5, 0, 0))
        camera = bpy.context.active_object
        texflow_props.camera = camera

        bpy.ops.mesh.primitive_ico_sphere_add()
        obj = bpy.context.active_object
        select_obj(obj)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.texflow.generate()

        kick_async_loop()
        kick_async_loop()

        bpy.ops.texflow.interrupt()

        limit = 100
        i = 0
        stop = False
        while not stop:
            stop = kick_async_loop()
            print(i, get_texflow_state().current_step)
            time.sleep(0.1)
            i += 1
            self.assertLess(i, limit)

        self.assertEqual(len(obj.data.materials), 0)
        self.assertFalse(get_texflow_state().status)

    def test_generate(self):
        state = get_texflow_state()
        pipe = load_pipe(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            controlnet_models_or_paths=["hf-internal-testing/tiny-controlnet"],
        )
        state.pipe = pipe
        texflow_props = bpy.context.scene.texflow
        height = 64
        width = 64
        texflow_props.height = height
        texflow_props.width = width
        texflow_props.steps = 4
        bpy.ops.object.camera_add(location=(0.0, -3.0, 0.0), rotation=(1.5, 0, 0))
        camera = bpy.context.active_object
        texflow_props.camera = camera

        bpy.ops.mesh.primitive_ico_sphere_add()
        obj = bpy.context.active_object
        select_obj(obj)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.texflow.generate()

        limit = 100
        i = 0
        stop = False
        while not stop:
            stop = kick_async_loop()
            print(i, get_texflow_state().current_step)
            time.sleep(0.1)
            i += 1
            self.assertLess(i, limit)

        new_material = obj.data.materials[-1]
        diffuse_node = next(
            node for node in new_material.node_tree.nodes if node.type == "TEX_IMAGE"
        )
        generated_image = diffuse_node.image

        self.assertEqual((height, width), tuple(generated_image.size))
        self.assertEqual("PENDING", get_texflow_state().status)

    def test_load_model(self):
        bpy.context.scene.texflow.current_model_path.model_path = (
            "hf-internal-testing/tiny-stable-diffusion-pipe"
        )
        bpy.context.scene.texflow.current_model_path.controlnet_model_path = (
            "hf-internal-testing/tiny-controlnet"
        )
        bpy.ops.texflow.load_model()
        timeout = 10
        st = time.time()
        while True:
            kick_async_loop()
            state = get_texflow_state()
            if state.pipe is not None:
                break
            duration = time.time() - st
            self.assertLess(duration, timeout)
            time.sleep(0.1)

        self.assertIsNotNone(get_texflow_state().pipe)
        self.assertGreater(get_texflow_state().load_step, 0)

    def test_spam_load_model(self):
        bpy.context.scene.texflow.current_model_path.model_path = (
            "hf-internal-testing/tiny-stable-diffusion-pipe"
        )
        bpy.context.scene.texflow.current_model_path.controlnet_model_path = (
            "hf-internal-testing/tiny-controlnet"
        )
        bpy.ops.texflow.load_model()
        bpy.ops.texflow.load_model()
        bpy.ops.texflow.load_model()
        bpy.ops.texflow.load_model()
        bpy.ops.texflow.load_model()
        timeout = 10
        st = time.time()
        while True:
            kick_async_loop()
            state = get_texflow_state()
            if state.pipe is not None:
                break
            duration = time.time() - st
            self.assertLess(duration, timeout)
            time.sleep(0.1)

        self.assertIsNotNone(state.pipe)

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

    def test_apply_history(self):
        texflow = bpy.context.scene.texflow
        self.assertEqual(0, len(texflow.model_path_history))

        texflow.current_model_path.model_path = (
            "hf-internal-testing/tiny-stable-diffusion-pipe"
        )
        texflow.current_model_path.controlnet_model_path = (
            "hf-internal-testing/tiny-controlnet"
        )
        bpy.ops.texflow.load_model()
        timeout = 10
        st = time.time()
        while True:
            kick_async_loop()
            state = get_texflow_state()
            if state.pipe is not None:
                break
            duration = time.time() - st
            self.assertLess(duration, timeout)
            time.sleep(0.1)

        self.assertIsNotNone(state.pipe)

        self.assertEqual(1, len(texflow.model_path_history))

        texflow.model_path_history_index = 0
        model_path_2 = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
        texflow.model_path_history[0].model_path = model_path_2
        texflow.model_path_history[0].controlnet_model_path = (
            "hf-internal-testing/tiny-controlnet-sdxl"
        )

        bpy.ops.texflow.apply_model_history()

        self.assertEqual(model_path_2, texflow.current_model_path.model_path)
