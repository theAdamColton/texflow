import bpy

from ..state import TexflowState
from ..utils import DESCRIPTION, VERSION_TUPLE
from .ui import (
    LoadModelOperator,
    TexflowProperties,
    TexflowPanel,
    StartGenerationOperator,
    StopGenerationOperator,
)
from .async_loop import AsyncLoopModalOperator


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
    TexflowPanel,
    StartGenerationOperator,
    StopGenerationOperator,
    TexflowProperties,
    AsyncLoopModalOperator,
    LoadModelOperator,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.texflow = bpy.props.PointerProperty(type=TexflowProperties)
    bpy.app.driver_namespace["texflow_state"] = TexflowState()


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.texflow
    del bpy.app.driver_namespace["texflow_state"]
