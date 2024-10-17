import random
import torch
import traceback
import numpy as np
import asyncio
import bpy
from bpy.props import StringProperty, IntProperty, BoolProperty, FloatProperty

from .camera import ensure_temp_camera
from .uv import uv_proj
from ..state import TexflowState
from .async_loop import AsyncModalOperatorMixin
from ..controller.pipe_utils import load_pipe, run_pipe
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
    model_path: bpy.props.StringProperty(
        name="Base model path",
        description="URL of pretrained model hosted inside a model repo on huggingface.co",
        default="stabilityai/stable-diffusion-2-1",
    )
    controlnet_model_path: bpy.props.StringProperty(
        name="Controlnet Path",
        description="URL of a pretrained controlnet model hosted inside a model repo on huggingface.co",
        default="thibaud/controlnet-sd21-depth-diffusers",
    )
    token: bpy.props.StringProperty(
        name="Huggingface Token",
        description="Your private huggingface API token",
        default="",
    )

    prompt: bpy.props.StringProperty(name="Prompt")
    negative_prompt: StringProperty(
        name="Negative Prompt",
        description="The model will avoid aspects of the negative prompt",
    )
    height: IntProperty(name="Height", default=512, min=64, step=64)
    width: IntProperty(name="Width", default=512, min=64, step=64)
    randomize_seed: BoolProperty(
        name="Randomize Seed",
        default=True,
        description="Randomly pick a seed every generation",
    )
    seed: IntProperty(
        name="Seed", default=42, description="Manual seed for random noise"
    )
    steps: IntProperty(name="Steps", default=25, min=1)
    camera: bpy.props.PointerProperty(
        name="Camera",
        type=bpy.types.Object,
        description="Render conditioning images from a camera",
    )
    cfg_scale: FloatProperty(
        name="CFG Scale",
        default=7.5,
        min=0,
        max=500.0,
        description="How strongly the prompt influences the image",
    )
    controlnet_conditioning_scale: FloatProperty(
        name="Controlnet Conditioning Scale",
        description="How strongly the controlnet effects the model",
        min=0.0,
        max=1.0,
        default=1.0,
    )
    image2image_strength: FloatProperty(
        name="Image2Image Strength",
        description="How strongly the image effects the model",
        min=0.0,
        max=1.0,
        default=0.6,
    )


class StartGenerationOperator(bpy.types.Operator, AsyncModalOperatorMixin):
    bl_label = "Generate"
    bl_idname = "texflow.generate"
    bl_description = "Generate a texture"

    stop_upon_exception = True

    @classmethod
    def poll(cls, context: bpy.types.Context | None):
        return (
            context is not None
            and context.mode == "EDIT_MESH"
            and context.active_object is not None
            and context.active_object.type == "MESH"
        )

    async def update_ui(self, step):
        texflow_state = get_texflow_state()
        texflow_state.current_step = step
        should_stop = not texflow_state.is_running
        return should_stop

    async def async_execute(self, context):
        texflow = context.scene.texflow
        get_texflow_state().is_running = True

        height = texflow.height
        width = texflow.width
        camera_obj = texflow.camera
        obj = context.active_object
        prompt = texflow.prompt

        if texflow.randomize_seed:
            max_abs = 2**31 - 1
            texflow.seed = random.randint(-max_abs, max_abs)

        loop = asyncio.get_event_loop()

        def callback_on_step_end(pipe, step, timestep, callback_kwargs):
            should_stop = loop.run_until_complete(self.update_ui(step))

            if should_stop:
                pipe._interrupt = True
                # Not all pipes implement _interrupt,
                # So we just quit
                raise Exception("Diffusion pipe interrupted")

            return callback_kwargs

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

        pipe = get_texflow_state().pipe

        # controlnet uses an inverted depth map with 3 color channels
        depth_map = 1 - depth_map
        depth_map = depth_map.unsqueeze(0).repeat(3, 1, 1)

        generated_image = await asyncio.to_thread(
            run_pipe,
            pipe=pipe,
            prompt=prompt,
            negative_prompt=texflow.negative_prompt,
            controlnet_conditioning_scales=[texflow.controlnet_conditioning_scale],
            image2image_strength=texflow.image2image_strength,
            control_images=[depth_map.unsqueeze(0)],
            height=height,
            width=width,
            num_inference_steps=texflow.steps,
            guidance_scale=texflow.cfg_scale,
            seed=texflow.seed,
            callback_on_step_end=callback_on_step_end,
        )

        should_stop = not get_texflow_state().is_running
        if should_stop:
            self.quit()
            return

        clean_prompt = "".join([c if c.isalnum() else "-" for c in prompt][:20])
        base_name = f"texflow{clean_prompt}"
        blender_image = bpy.data.images.new(
            f"{base_name}_Texture",
            width=width,
            height=height,
            alpha=False,
        )
        # need to add an alpha channel for converting to blender
        generated_image = np.concatenate(
            (
                generated_image,
                np.zeros((height, width, 1), dtype=generated_image.dtype),
            ),
            axis=-1,
        )
        blender_image.pixels.foreach_set(generated_image.flatten())
        blender_image.pack()
        blender_image.reload()

        material = bpy.data.materials.new(name=f"{base_name}_Material")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        for node in nodes:
            nodes.remove(node)

        output_node = nodes.new(type="ShaderNodeOutputMaterial")
        output_node.location = (300, 0)
        bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf_node.location = (0, 0)
        bsdf_node.inputs["Metallic"].default_value = 0.0
        bsdf_node.inputs["Roughness"].default_value = 1.0
        diffuse_map_node = nodes.new(type="ShaderNodeTexImage")
        diffuse_map_node.location = (-600, 300)
        diffuse_map_node.image = blender_image
        uv_map_node = nodes.new(type="ShaderNodeUVMap")
        uv_map_node.location = (-800, 500)
        uv_map_node.uv_map = uv_layer.name

        links.new(uv_map_node.outputs["UV"], diffuse_map_node.inputs["Vector"])
        links.new(diffuse_map_node.outputs["Color"], bsdf_node.inputs["Base Color"])
        links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

        obj.data.materials.append(material)
        material_index = obj.data.materials.find(material.name)
        mesh = obj.data
        for p in mesh.polygons:
            if p.select:
                p.material_index = material_index

        print("GENERATED IMAGE TO", blender_image.name)

        get_texflow_state().is_running = False

        self.quit()


class LoadModelOperator(bpy.types.Operator, AsyncModalOperatorMixin):
    bl_label = "Load model"
    bl_idname = "texflow.load_model"
    bl_description = "Load a model from huggingface"

    @classmethod
    def poll(cls, context):
        texflow = context.scene.texflow
        return texflow.model_path != ""

    async def async_execute(self, context):
        texflow = context.scene.texflow
        try:
            pipe = await asyncio.to_thread(
                load_pipe,
                pretrained_model_or_path=texflow.model_path,
                controlnet_models_or_paths=[texflow.controlnet_model_path],
                token=texflow.token if texflow.token != "" else None,
            )
        except Exception as e:
            traceback.print_exception(e)
            self.report({"ERROR"}, str(e))
            self.quit()
            return

        print("Loaded Pipe")

        state = get_texflow_state()
        state.pipe = pipe

        ui_update(None, context)

        self.quit()


class StopGenerationOperator(bpy.types.Operator):
    bl_label = "Interrupt"
    bl_idname = "texflow.interrupt"
    bl_description = "Stop generation"

    def execute(self, context):
        get_texflow_state().is_running = False
        return {"FINISHED"}


class _TexflowPanelMixin:
    bl_category = "texflow"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"


class TexflowParentPanel(bpy.types.Panel, _TexflowPanelMixin):
    bl_label = "texflow"
    bl_idname = "TEXFLOW_PT_texflow_parent_panel"

    def draw(self, context):
        pass


class TexflowModelPanel(bpy.types.Panel, _TexflowPanelMixin):
    bl_label = "Model"
    bl_idname = "TEXFLOW_PT_texflow_model_panel"
    bl_parent_id = TexflowParentPanel.bl_idname

    def draw(self, context):
        layout = self.layout
        texflow_state = get_texflow_state()
        texflow = context.scene.texflow

        is_running = texflow_state.is_running
        pipe = texflow_state.pipe
        model_loaded = pipe is not None

        layout.label(text="Base Model Path:")
        layout.prop(texflow, "model_path", text="")
        layout.label(text="ControlNet Model path (Optional):")
        layout.prop(texflow, "controlnet_model_path", text="")
        layout.label(text="HuggingFace token (Optional):")
        layout.prop(texflow, "token", text="")
        layout.operator(LoadModelOperator.bl_idname)


class TexflowAdvancedPromptPanel(bpy.types.Panel, _TexflowPanelMixin):
    bl_label = "Advanced Prompting"
    bl_idname = "TEXFLOW_PT_texflow_avanced_prompt_panel"
    bl_parent_id = TexflowParentPanel.bl_idname

    def draw(self, context):
        layout = self.layout
        texflow = context.scene.texflow

        layout.prop(texflow, "seed")
        layout.prop(texflow, "cfg_scale")
        layout.prop(texflow, "steps")
        layout.label(text="Negative Prompt:")
        layout.prop(texflow, "negative_prompt", text="")
        layout.prop(texflow, "controlnet_conditioning_scale")
        layout.prop(texflow, "image2image_strength")


class TexflowPromptPanel(bpy.types.Panel, _TexflowPanelMixin):
    bl_label = "Prompting"
    bl_idname = "TEXFLOW_PT_texflow_prompt_panel"
    bl_parent_id = TexflowParentPanel.bl_idname
    bl_options = {"HIDE_HEADER"}

    def draw(self, context):
        layout = self.layout
        texflow_state = get_texflow_state()
        texflow = context.scene.texflow

        is_running = texflow_state.is_running
        pipe = texflow_state.pipe
        model_loaded = pipe is not None

        layout.prop_search(
            texflow, "camera", bpy.data, "objects", text="Camera (Optional)"
        )

        if model_loaded:
            layout.label(text=f"Model loaded: {pipe.name_or_path}")
        else:
            layout.label(text=f"No model loaded.")

        layout.label(text="Prompt:")
        layout.prop(texflow, "prompt", text="")

        layout.prop(texflow, "height")
        layout.prop(texflow, "width")

        row = layout.row()
        row.enabled = (not is_running) and model_loaded
        row.operator(StartGenerationOperator.bl_idname)

        row = layout.row()
        row.progress(
            text=f"{texflow_state.current_step}/{texflow.steps}",
            factor=texflow_state.current_step / texflow.steps,
        )
        row.operator(StopGenerationOperator.bl_idname, text="Interrupt")
        row.enabled = is_running
