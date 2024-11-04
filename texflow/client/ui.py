import json
import aiohttp
import io
import bpy

from .utils import to_tiff
from .camera import ensure_temp_camera
from .uv import uv_proj
from ..state import TexflowState
from .async_loop import AsyncModalOperatorMixin
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
        description="URL of ComfyUI server",
        default="127.0.0.1:8188",
    )


class RenderDepthImageOperator(bpy.types.Operator, AsyncModalOperatorMixin):
    bl_label = "RenderDepthImage"
    bl_idname = "texflow.render_depth_image"
    bl_description = "Render a depth image and send it to comfyui"
    task_name = "texflow.render_depth_image"

    stop_upon_exception = True

    height: bpy.props.IntProperty(
        description="Height of generated depth map", min=16, max=8192, default=512
    )
    width: bpy.props.IntProperty(
        description="Width of generated depth map", min=16, max=8192, default=512
    )

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
        height = self.height
        width = self.width
        camera_obj = texflow.camera
        obj = context.active_object

        with ensure_temp_camera(camera_obj) as camera_obj:
            depth_map, depth_occupancy = render_depth_map(
                obj=obj,
                camera_obj=camera_obj,
                height=height,
                width=width,
            )
            uv_layer = uv_proj(
                obj=obj,
                camera_obj=camera_obj,
                height=height,
                width=width,
            )

        # controlnet uses an inverted depth map
        depth_map = 1 - depth_map
        depth_image = to_tiff(depth_map)

        depth_image_bytes = io.BytesIO()
        depth_image.save(depth_image_bytes)
        depth_image_bytes.seek(0)

        post_data = {
            "image": {
                "file": depth_image_bytes.read(),
                "filename": "texflowDepthImage.tiff",
            },
            "overwrite": "true",
        }
        image_post_url = texflow.url + "/upload/image"

        async with aiohttp.ClientSession(timeout=5) as sess:
            response = await sess.post(image_post_url, json=json.dumps(post_data))
            print("GOT REPONSE", response)

        self.quit()


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

        texflow_status = texflow_state.status

        layout.separator()
        layout.prop_search(texflow, "camera", bpy.data, "objects")

        layout.label(text="Prompt:")
        layout.prop(texflow, "prompt", text="")

        layout.prop(texflow, "height")
        layout.prop(texflow, "width")

        layout.separator()
        row = layout.row()
        row.operator(StartGenerationOperator.bl_idname)

        row = layout.row()
        row.progress(
            text=f"{texflow_state.current_step}/{texflow.steps}",
            factor=texflow_state.current_step / texflow.steps,
        )
        row.operator(InterruptGenerationOperator.bl_idname, text="Interrupt")
