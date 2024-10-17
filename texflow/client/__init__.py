import torch
import gc
import asyncio
import bpy

from ..state import TexflowState
from ..utils import DESCRIPTION, VERSION_TUPLE
from .ui import (
    LoadModelOperator,
    TexflowProperties,
    TexflowParentPanel,
    TexflowPromptPanel,
    TexflowModelPanel,
    TexflowAdvancedPromptPanel,
    StartGenerationOperator,
    StopGenerationOperator,
)
from .async_loop import AsyncLoopModalOperator, AsyncModalOperatorMixin


bl_info = {
    "name": "texflow",
    "author": "Adam Colton",
    "description": DESCRIPTION,
    "blender": (4, 2, 1),
    "version": VERSION_TUPLE,
    "location": "Image Editor -> Sidebar -> texflow",
    "category": "Paint",
}

classes = (
    TexflowParentPanel,
    TexflowModelPanel,
    TexflowAdvancedPromptPanel,
    TexflowPromptPanel,
    StartGenerationOperator,
    StopGenerationOperator,
    TexflowProperties,
    AsyncLoopModalOperator,
    LoadModelOperator,
)


def _stop_all_tasks():
    try:
        tasks = asyncio.all_tasks()
    except:
        return
    for task in tasks:
        if task.get_name().startswith("texflow"):
            task.cancel()


def register():
    _stop_all_tasks()
    bpy.app.driver_namespace["texflow_state"] = TexflowState()
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.texflow = bpy.props.PointerProperty(type=TexflowProperties)


def unregister():
    _stop_all_tasks()
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.texflow
    del bpy.app.driver_namespace["texflow_state"]
    # this is necessary to clear up the memory used by the diffusion model
    gc.collect()
    torch.cuda.empty_cache()
