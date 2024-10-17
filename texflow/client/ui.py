import random
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


class ModelPathProperty(bpy.types.PropertyGroup):
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


class TexflowProperties(bpy.types.PropertyGroup):
    model_path_history: bpy.props.CollectionProperty(type=ModelPathProperty)
    model_path_history_index: bpy.props.IntProperty(default=0)
    current_model_path: bpy.props.PointerProperty(type=ModelPathProperty)
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
    task_name = "texflow.generate"

    stop_upon_exception = True

    @classmethod
    def poll(cls, context: bpy.types.Context | None):
        return (
            context is not None
            and context.mode == "EDIT_MESH"
            and context.active_object is not None
            and context.active_object.type == "MESH"
        )

    async def async_execute(self, context):
        texflow_state = get_texflow_state()
        if texflow_state.status != "PENDING":
            raise ValueError(
                f"Can't start generation when texflow_status is {texflow_state.status}!"
            )

        get_texflow_state().status = "RUNNING"
        try:
            await self._generate(context)
        finally:
            get_texflow_state().status = "PENDING"
            self.quit()

    async def _generate(self, context):
        texflow = context.scene.texflow
        texflow_state = get_texflow_state()

        height = texflow.height
        width = texflow.width
        camera_obj = texflow.camera
        obj = context.active_object
        prompt = texflow.prompt

        if texflow.randomize_seed:
            max_abs = 2**31 - 1
            texflow.seed = random.randint(-max_abs, max_abs)

        loop = asyncio.get_event_loop()

        async def _update_ui_main_thread(step):
            get_texflow_state().current_step = step
            ui_update(None, bpy.context)

        def callback_on_step_end(_, step, timestep, callback_kwargs):
            asyncio.run_coroutine_threadsafe(
                asyncio.to_thread(_update_ui_main_thread, step), loop
            )
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
        self.report({"INFO"}, "Finished generating image")

        get_texflow_state().status = "PENDING"


class LoadModelOperator(bpy.types.Operator, AsyncModalOperatorMixin):
    bl_label = "Load model"
    bl_idname = "texflow.load_model"
    bl_description = "Load a model from huggingface"
    task_name = "texflow.load_model"

    @classmethod
    def poll(cls, context):
        texflow = context.scene.texflow
        return texflow.current_model_path.model_path != ""

    async def async_execute(self, context):
        texflow_state = get_texflow_state()
        if texflow_state.status != "PENDING":
            raise ValueError(
                f"Can't load a model when texflow_status is {texflow_state.status}!"
            )

        get_texflow_state().status = "LOADING"
        try:
            await self._load_model(context)
        finally:
            get_texflow_state().status = "PENDING"
            self.quit()

    async def _load_model(self, context):
        texflow = context.scene.texflow
        get_texflow_state().pipe = None

        model_path = texflow.current_model_path.model_path
        controlnet_model_path = texflow.current_model_path.controlnet_model_path

        main_thread_loop = asyncio.get_event_loop()

        def _update_ui_mainthread(n, total):
            get_texflow_state().load_step = n
            get_texflow_state().load_max_step = total
            ui_update(None, bpy.context)
            print("UI UPDATE MAIN THREAD", n, total)

        def tqdm_callback(n, total):
            print("TQDM CALLBACK", n, total)
            asyncio.run_coroutine_threadsafe(
                asyncio.to_thread(_update_ui_mainthread, n, total), main_thread_loop
            )

        try:
            pipe = await asyncio.to_thread(
                load_pipe,
                pretrained_model_or_path=model_path,
                controlnet_models_or_paths=[controlnet_model_path],
                token=texflow.token if texflow.token != "" else None,
                tqdm_callback=tqdm_callback,
            )
        except Exception as e:
            get_texflow_state().status = "PENDING"
            traceback.print_exception(e)
            self.report({"ERROR"}, str(e))
            return

        get_texflow_state().load_step = get_texflow_state().load_max_step
        ui_update(None, bpy.context)

        # removes identical items from history
        for index, saved_model_path_property in enumerate(texflow.model_path_history):
            saved_model_path = saved_model_path_property.model_path
            saved_controlnet_path = saved_model_path_property.controlnet_model_path
            if (
                saved_model_path == model_path
                and saved_controlnet_path == controlnet_model_path
            ):
                texflow.model_path_history.remove(index)

        # adds to history
        saved_model_path_property = texflow.model_path_history.add()
        saved_model_path_property.model_path = model_path
        saved_model_path_property.controlnet_model_path = controlnet_model_path

        self.report({"INFO"}, f"Loaded Pipe {pipe.name_or_path}")

        state = get_texflow_state()
        state.pipe = pipe

        ui_update(None, context)

        get_texflow_state().status = "PENDING"


class StopGenerationOperator(bpy.types.Operator):
    bl_label = "Interrupt"
    bl_idname = "texflow.interrupt"
    bl_description = "Stop generation"

    def execute(self, context):
        StartGenerationOperator.cancel_tasks()
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


class TexflowApplyModelHistory(bpy.types.Operator):
    bl_label = "Use"
    bl_idname = "texflow.apply_model_history"
    bl_description = "Use this model path"

    @classmethod
    def poll(cls, context):
        texflow = context.scene.texflow
        index = texflow.model_path_history_index
        return index >= 0 and index < len(texflow.model_path_history)

    def execute(self, context):
        texflow = context.scene.texflow
        model_path_property = texflow.model_path_history[
            texflow.model_path_history_index
        ]

        texflow.current_model_path.model_path = model_path_property.model_path
        texflow.current_model_path.controlnet_model_path = (
            model_path_property.controlnet_model_path
        )

        return {"FINISHED"}

        self.model_path_property


class TexflowModelHistoryList(bpy.types.UIList):
    bl_idname = "TEXFLOW_UL_texflow_model_history_list"

    def draw_item(
        self,
        context,
        layout,
        data,
        item,
        icon,
        active_data,
        active_property,
        index=0,
        flt_flag=0,
    ):
        text = item.model_path
        if item.controlnet_model_path != "":
            text += f"  {item.controlnet_model_path}"
        row = layout.row()
        row.label(text=text)


class TexflowModelPanel(bpy.types.Panel, _TexflowPanelMixin):
    bl_label = "Model"
    bl_idname = "TEXFLOW_PT_texflow_model_panel"
    bl_parent_id = TexflowParentPanel.bl_idname

    def draw(self, context):
        layout = self.layout
        texflow_state = get_texflow_state()
        texflow = context.scene.texflow

        texflow_status = texflow_state.status
        pipe = texflow_state.pipe
        model_loaded = pipe is not None

        layout.label(text="Base Model Path:")
        layout.prop(texflow.current_model_path, "model_path", text="")
        layout.label(text="ControlNet Model path (Optional):")
        layout.prop(texflow.current_model_path, "controlnet_model_path", text="")
        layout.label(text="HuggingFace token (Optional):")
        layout.prop(texflow, "token", text="")

        layout.separator()
        layout.label(text="Model History:")
        layout.template_list(
            TexflowModelHistoryList.bl_idname,
            "",
            texflow,
            "model_path_history",
            texflow,
            "model_path_history_index",
        )
        layout.operator(TexflowApplyModelHistory.bl_idname)

        layout.separator(factor=1.5)
        row = layout.row()
        col = row.column()
        col.operator(LoadModelOperator.bl_idname)
        col.enabled = texflow_status == "PENDING"

        col = row.column()
        col.progress(
            text=f"{texflow_state.load_step}/{texflow_state.load_max_step}",
            factor=texflow_state.load_step / max(texflow_state.load_max_step, 1),
        )
        col.enabled = texflow_status == "LOADING"


class TexflowAdvancedPromptPanel(bpy.types.Panel, _TexflowPanelMixin):
    bl_label = "Advanced Prompting"
    bl_idname = "TEXFLOW_PT_texflow_avanced_prompt_panel"
    bl_parent_id = TexflowParentPanel.bl_idname
    bl_options = {"DEFAULT_CLOSED"}

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

        texflow_status = texflow_state.status
        pipe = texflow_state.pipe
        model_loaded = pipe is not None

        if model_loaded:
            layout.label(text=f"Model loaded: {pipe.name_or_path}")
        else:
            layout.label(text=f"No model loaded.")

        layout.separator()
        layout.prop_search(texflow, "camera", bpy.data, "objects")

        layout.label(text="Prompt:")
        layout.prop(texflow, "prompt", text="")

        layout.prop(texflow, "height")
        layout.prop(texflow, "width")

        layout.separator()
        row = layout.row()
        row.enabled = (texflow_status == "PENDING") and model_loaded
        row.operator(StartGenerationOperator.bl_idname)

        row = layout.row()
        row.progress(
            text=f"{texflow_state.current_step}/{texflow.steps}",
            factor=texflow_state.current_step / texflow.steps,
        )
        row.operator(StopGenerationOperator.bl_idname, text="Interrupt")
        row.enabled = texflow_status == "RUNNING"
