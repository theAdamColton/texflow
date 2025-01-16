import numpy as np
import logging
import uuid
from PIL.PngImagePlugin import PngInfo
from PIL import Image
import aiohttp
import io
import bpy

from .utils import to_image16
from .camera import ensure_temp_camera
from .uv import uv_proj
from ..state import TexflowState
from .async_loop import AsyncLoopManager, AsyncModalOperatorMixin
from .depth import render_depth_map


def get_texflow_state():
    state: TexflowState = bpy.app.driver_namespace["texflow_state"]
    return state


def ui_update(_, context):
    """
    https://blender.stackexchange.com/questions/238441/force-redraw-add-on-custom-propery-in-n-panel-from-a-separate-thread
    """
    if context.area is not None:
        for region in context.area.regions:
            if region.type == "UI":
                region.tag_redraw()


class TexflowProperties(bpy.types.PropertyGroup):
    camera: bpy.props.PointerProperty(
        name="Camera",
        type=bpy.types.Object,
        description="The camera from which to capture the depth image",
    )
    comfyui_url: bpy.props.StringProperty(
        name="URL",
        description="URL of the ComfyUI server",
        default="127.0.0.1:8188",
    )


class TexflowAsyncOperator(AsyncModalOperatorMixin):
    async_loop_manager_name = "TexflowAsyncLoop"

    @staticmethod
    def get_async_manager():
        return AsyncLoopManager.register(TexflowAsyncOperator.async_loop_manager_name)


class RenderDepthImageOperator(TexflowAsyncOperator, bpy.types.Operator):
    bl_label = "Render Depth Image"
    bl_idname = "texflow.render_depth_image"
    bl_description = "Renders a depth image and sends it to comfyui"

    @classmethod
    def poll(cls, context):
        texflow = context.scene.texflow
        return (
            context is not None
            and context.mode == "EDIT_MESH"
            and context.active_object is not None
            and context.active_object.type == "MESH"
            and texflow.camera is not None
        )

    async def async_execute(self, context):
        texflow_state = get_texflow_state()
        texflow = context.scene.texflow
        camera_obj = texflow.camera
        obj = context.active_object

        render_id = str(uuid.uuid4())
        texflow_state.render_id = render_id

        with ensure_temp_camera(camera_obj) as camera_obj:
            depth_map, depth_occupancy = render_depth_map(
                obj=obj,
                camera_obj=camera_obj,
            )
            uv_layer = uv_proj(
                obj=obj,
                camera_obj=camera_obj,
            )

        uv_layer.name = f"TexflowUVLayer-{render_id}"

        # controlnet uses an inverted depth map
        depth_map = 1 - depth_map
        depth_image = to_image16(depth_map)

        depth_image_bytes = io.BytesIO()
        depth_image.save(depth_image_bytes, format="tiff")
        depth_image_bytes = depth_image_bytes.getvalue()

        # save the render_id to the png image
        depth_occupancy_metadata = PngInfo()
        depth_occupancy_metadata.add_text("render_id", render_id)
        depth_occupancy = Image.fromarray(depth_occupancy)
        depth_occupancy_bytes = io.BytesIO()
        depth_occupancy.save(
            depth_occupancy_bytes, format="png", pnginfo=depth_occupancy_metadata
        )
        depth_occupancy_bytes = depth_occupancy_bytes.getvalue()

        async def post_image_bytes(_bytes, filename, content_type="image/tiff"):
            form_data = aiohttp.FormData()
            form_data.add_field(
                "image", _bytes, filename=filename, content_type=content_type
            )
            form_data.add_field("overwrite", "true")
            image_post_url = f"http://{texflow.comfyui_url}/upload/image"
            logging.info(f"Posting image to {image_post_url}")

            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.post(image_post_url, data=form_data) as response:
                        logging.info(f"Got post response {response}")
                        result = await response.json()
                        logging.info(f"Got post result {result}")
            except Exception as e:
                self.report({"WARNING"}, f"Error sending depth image to ComfyUI {e}")
                raise e

        await post_image_bytes(
            depth_image_bytes, "texflow_depth_image.tiff", content_type="image/tiff"
        )
        await post_image_bytes(
            depth_occupancy_bytes,
            "texflow_occupancy_image.png",
            content_type="image/png",
        )

        texflow_state.active_obj = obj

        self.report({"INFO"}, f"Sent depth image to ComfyUI")
        logging.info(f"Sent depth image to ComfyUI with render_id {render_id}")


class LoadComfyUIImages(TexflowAsyncOperator, bpy.types.Operator):
    bl_label = "Load ComfyUI Images"
    bl_idname = "texflow.load_comfyui_images"
    bl_description = "Load images from the ComfyUI server and apply them to your mesh"

    async def async_execute(self, context):
        texflow = context.scene.texflow
        texflow_state = get_texflow_state()
        obj = texflow_state.active_obj
        render_id = texflow_state.render_id

        history_url = f"http://{texflow.comfyui_url}/history"
        logging.info(f"Getting history from {history_url}")

        try:
            async with aiohttp.ClientSession() as sess:
                response = await sess.get(history_url)
                history_data = await response.json()
        except Exception as e:
            self.report({"WARNING"}, f"Error getting images from ComfyUI {e}")
            raise e

        image_filenames = []
        # if the render_id is the same, fetches the image
        for promt_id, generation_data in history_data.items():
            all_output_data = generation_data["outputs"]
            for output_data in all_output_data.values():
                output_images = output_data["images"]
                for image_data in output_images:
                    filename = image_data["filename"]
                    # all filenames should look like this:
                    # "texflow_e4f3f29c-6c50-4da5-b06d-b7687c648bbd_00003_.png"
                    try:
                        image_render_id = filename.split("_")[1]
                    except:
                        image_render_id = ""

                    logging.info(
                        f"got image with render id {image_render_id} {render_id}"
                    )
                    if filename.startswith("texflow") and image_render_id == render_id:
                        image_filenames.append(filename)

        if len(image_filenames) <= 0:
            self.report(
                {"WARNING"},
                "No suitable outputs found in ComfyUI. Have you tried saving a texflow texture using the custom node?",
            )
            raise ValueError(f"No images to download for render {render_id}")

        logging.info(f"Got {len(image_filenames)} for depth image {render_id}")

        uv_layer_name = f"TexflowUVLayer-{render_id}"
        uv_layer = obj.data.uv_layers[uv_layer_name]

        for image_filename in image_filenames:
            instance_name = image_filename.removesuffix(".png")

            image_url = f"http://{texflow.comfyui_url}/view?filename={image_filename}"
            logging.info(f"Getting from {image_url}")

            async with aiohttp.ClientSession() as sess:
                response = await sess.get(image_url)
                data = await response.read()

            pixel_values = np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))
            # flip height
            pixel_values = pixel_values[::-1]
            pixel_values = pixel_values.astype(np.float32) / 255
            height, width, _ = pixel_values.shape
            print(pixel_values.shape)
            image = bpy.data.images.new(image_filename, width=width, height=height)
            image.pixels = pixel_values.flatten()

            material = bpy.data.materials.new(name=instance_name)
            material.use_nodes = True

            # Create a new texture node and assign the image
            nodes = material.node_tree.nodes
            links = material.node_tree.links

            # Clear default nodes
            for node in nodes:
                nodes.remove(node)

            # Create BSDF node
            bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
            bsdf.location = (0, 0)

            # Create Image Texture node
            texture = nodes.new(type="ShaderNodeTexImage")
            texture.image = image
            texture.location = (-300, 0)

            # Create Output node
            output = nodes.new(type="ShaderNodeOutputMaterial")
            output.location = (300, 0)

            # Link nodes
            links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
            links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

        obj.data.uv_layers.active = uv_layer
        uv_layer.active_render = True
        for slot in obj.material_slots:
            slot.material = material

        self.report({"INFO"}, f"Added {len(image_filenames)} new generated materials")


class TexflowPanel(bpy.types.Panel):
    bl_category = "texflow"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    bl_label = "texflow"
    bl_idname = "TEXFLOW_PT_texflow_panel"

    def draw(self, context):
        layout = self.layout
        texflow_state = get_texflow_state()
        texflow = context.scene.texflow

        layout.label(text="ComfyUI Settings:")
        layout.prop(texflow, "comfyui_url")

        layout.separator(factor=2)

        layout.prop_search(texflow, "camera", bpy.data, "objects")
        layout.operator(RenderDepthImageOperator.bl_idname)
        layout.operator(LoadComfyUIImages.bl_idname)
