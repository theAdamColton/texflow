import uuid
import aiohttp
import io
import bpy

from .utils import to_image16
from .camera import ensure_temp_camera
from .uv import uv_proj
from ..state import TexflowState, TexflowStatus
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

    height: bpy.props.IntProperty(
        description="Height of generated depth map",
        min=16,
        max=8192,
        default=512,
        name="Height",
    )
    width: bpy.props.IntProperty(
        description="Width of generated depth map",
        min=16,
        max=8192,
        default=512,
        name="Width",
    )


class TexflowAsyncOperator(AsyncModalOperatorMixin):
    async_loop_manager_name = "TexflowAsyncLoop"

    @staticmethod
    def get_async_manager():
        return AsyncLoopManager.register(TexflowAsyncOperator.async_loop_manager_name)


class TexflowConnectToComfyOperator(TexflowAsyncOperator, bpy.types.Operator):
    bl_label = "Connect to ComfyUI"
    bl_idname = "texflow.connect_to_comfy"
    bl_description = "Connect to ComfyUI"

    async def async_execute(self, context):
        texflow_state = get_texflow_state()
        assert texflow_state.status != TexflowStatus.CONNECTING
        texflow = context.scene.texflow

        client_id = str(uuid.uuid4())
        ws_comfyui_url = f"ws://{texflow.comfyui_url}/ws?clientId={client_id}"
        self.logger.info(f"Connecting to {ws_comfyui_url}")

        texflow_state.status = TexflowStatus.CONNECTING

        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.ws_connect(ws_comfyui_url) as ws:
                    json = await ws.receive_json()
                    self.logger.info(f"Connected with json response {json}")

                    texflow_state.status = TexflowStatus.READY
                    ui_update(None, context)
                    texflow_state.client_id = client_id

                    async for msg in ws:
                        self.logger.info(f"Recieved ws msg {msg}")
        finally:
            texflow_state.status = TexflowStatus.PENDING


class RenderDepthImageOperator(TexflowAsyncOperator, bpy.types.Operator):
    bl_label = "Render Depth Image"
    bl_idname = "texflow.render_depth_image"
    bl_description = "Render a depth image and send it to comfyui"

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
        height = texflow.height
        width = texflow.width
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
        depth_image = to_image16(depth_map)

        depth_image_bytes = io.BytesIO()
        depth_image.save(depth_image_bytes, format="tiff")
        depth_image_bytes = depth_image_bytes.getvalue()

        form_data = aiohttp.FormData()
        form_data.add_field(
            "image",
            depth_image_bytes,
            filename="texflow_depth_image.tiff",
            content_type="image/tiff",
        )
        form_data.add_field("overwrite", "true")

        image_post_url = f"http://{texflow.comfyui_url}/upload/image"

        self.logger.info(f"Posting depth image to {image_post_url}")

        async with aiohttp.ClientSession() as sess:
            async with sess.post(image_post_url, data=form_data) as response:
                self.logger.info(f"Got post response {response}")
                result = await response.json()
                self.logger.info(f"Got post result {result}")

        print("RENDER DEPTH IMAGE DONE!")


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

        layout.label(text="ComfyUI Settings:")
        layout.prop(texflow, "comfyui_url")

        if texflow_state.status == TexflowStatus.PENDING:
            label_text = "Not connected."
        elif texflow_state.status == TexflowStatus.CONNECTING:
            label_text = "Connecting..."
        elif texflow_state.status == TexflowStatus.READY:
            label_text = "Connected."
        layout.label(text=label_text)

        row = layout.row()
        row.enabled = texflow_state.status != TexflowStatus.CONNECTING
        row.operator(TexflowConnectToComfyOperator.bl_idname)

        layout.separator(factor=2)

        layout.prop_search(texflow, "camera", bpy.data, "objects")
        layout.prop(texflow, "height")
        layout.prop(texflow, "width")
        row = layout.row()
        row.operator(RenderDepthImageOperator.bl_idname)
