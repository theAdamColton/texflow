import numpy as np
import asyncio
import bpy
from bpy.props import StringProperty, IntProperty, BoolProperty, FloatProperty

from texflow.client.uv import uv_proj
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


class StartGenerationOperator(bpy.types.Operator, AsyncModalOperatorMixin):
    bl_label = "TEXFLOW_OT_Generate"
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
        bpy.context.scene.texflow.current_step = step
        should_stop = not bpy.context.scene.texflow.is_running
        return should_stop

    async def async_execute(self, context):
        texflow = context.scene.texflow
        texflow.is_running = True

        height = texflow.height
        width = texflow.width
        camera = texflow.camera
        obj = context.active_object
        prompt = texflow.prompt

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
        depth_map, depth_occupancy = render_depth_map(
            obj=obj,
            camera_obj=camera,
            height=height,
            width=width,
        )
        uv_layer = uv_proj(
            obj=obj,
            camera_obj=camera,
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

        should_stop = not texflow.is_running
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

        texflow.is_running = False

        self.quit()


class StopGenerationOperator(bpy.types.Operator):
    bl_label = "TEXFLOW_OT_Interrupt"
    bl_idname = "texflow.interrupt"
    bl_description = "Stop generation"

    def execute(self, context):
        context.scene.texflow.is_running = False
        return {"FINISHED"}


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
        layout.prop()

        row = layout.row()
        row.enabled = not is_running
        row.operator(StartGenerationOperator.bl_idname)

        row = layout.row()
        row.progress(
            text=f"{texflow.current_step}/{texflow.steps}",
            factor=texflow.current_step / texflow.steps,
        )
        row.operator(StopGenerationOperator.bl_idname, text="Interrupt")
        row.enabled = is_running
