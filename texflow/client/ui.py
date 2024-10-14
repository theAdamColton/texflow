import asyncio
import bpy
from bpy.props import StringProperty, IntProperty, BoolProperty, FloatProperty

from texflow.state import TexflowState

from .async_loop import AsyncModalOperatorMixin
from ..controller.pipe_utils import run_pipe
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
    prompt: bpy.props.StringProperty(name="Prompt")
    negative_prompt: StringProperty(
        name="Negative Prompt",
        description="The model will avoid aspects of the negative prompt",
    )
    height: IntProperty(name="Height", default=512, min=64, step=64)
    width: IntProperty(name="Width", default=512, min=64, step=64)
    random_seed: BoolProperty(
        name="Random Seed", default=True, description="Randomly pick a seed"
    )
    seed: StringProperty(name="Seed", default="0", description="Manually pick a seed")
    steps: IntProperty(name="Steps", default=25, min=1)
    is_running: BoolProperty(default=False)
    current_step: IntProperty(default=0, update=ui_update)
    camera: bpy.props.PointerProperty(
        name="Camera",
        type=bpy.types.Object,
        description="Render conditioning images from a camera",
    )
    cfg_scale: FloatProperty(
        name="CFG Scale",
        default=7.5,
        min=0,
        description="How strongly the prompt influences the image",
    )
    controlnet_conditioning_scale: FloatProperty(
        name="Controlnet Conditioning Scale",
        description="How strongly the controlnet effects the model",
        min=0.0,
        max=1.0,
    )
    image2image_strength: FloatProperty(
        name="Image2Image Strength",
        description="How strongly the image effects the model",
        min=0.0,
        max=1.0,
    )


class TEXFLOW_OT_Generate(bpy.types.Operator, AsyncModalOperatorMixin):
    bl_label = "TEXFLOW_OT_Generate"
    bl_idname = "texflow.generate"
    bl_description = "Generate a texture"

    def update_ui(self, step):
        print("UPDATE CURRENT STEP", step)
        bpy.context.state.texflow.current_step = step

    async def async_execute(self, context):
        print("STARTING GENERATION")
        texflow = context.scene.texflow

        def callback_on_step_end(pipe, step, timestep, callback_kwargs):
            texflow.current_step = step
            asyncio.run_coroutine_threadsafe(self.update_ui, asyncio.get_event_loop())

        texflow.is_running = True
        """
        depth_map = await asyncio.to_thread(
            render_depth_map,
            obj=context.active_object,
            camera=texflow.camera,
            height=texflow.height,
            width=texflow.width,
        )
        """
        depth_map, depth_occupancy = render_depth_map(
            obj=context.active_object,
            camera_obj=texflow.camera,
            height=texflow.height,
            width=texflow.width,
        )

        pipe = get_texflow_state().pipe

        generated_image = await asyncio.to_thread(
            run_pipe,
            pipe=pipe,
            prompt=texflow.prompt,
            negative_prompt=texflow.negative_prompt,
            controlnet_conditioning_scales=[texflow.controlnet_conditioning_scale],
            image2image_strength=texflow.image2image_strength,
            control_images=[depth_map.unsqueeze(0)],
            height=texflow.height,
            width=texflow.width,
            num_inference_steps=texflow.steps,
            guidance_scale=texflow.cfg_scale,
            seed=texflow.seed,
            callback_on_step_end=callback_on_step_end,
        )
        texflow.is_running = False
        print("GENERATED IMAGE", generated_image)
        self.quit()


class TexflowPanel(bpy.types.Panel):
    bl_label = "texflow"
    bl_idname = f"TEXFLOW_PT_texflow_panel_IMAGE_EDITOR"
    bl_category = "texflow"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        texflow = context.scene.texflow

        is_running = texflow.is_running

        layout.prop_search(texflow, "camera", bpy.data, "objects")
        row = layout.row()
        row.progress(
            text=f"{texflow.current_step}/{texflow.steps}",
            factor=texflow.current_step / texflow.steps,
        )
