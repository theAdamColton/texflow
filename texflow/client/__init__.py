import bpy
import logging

from ..state import TexflowState
from ..utils import DESCRIPTION, VERSION_TUPLE
from .ui import (
    TexflowProperties,
    TexflowPanel,
    RenderDepthImageOperator,
    TexflowAsyncOperator,
    ConnectToComfyOperator,
)


logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
)

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
    RenderDepthImageOperator,
    TexflowProperties,
    ConnectToComfyOperator,
)


def register():
    async_loop_mgr = TexflowAsyncOperator.get_async_manager()
    async_loop_mgr.unregister()
    bpy.app.driver_namespace["texflow_state"] = TexflowState()
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.texflow = bpy.props.PointerProperty(type=TexflowProperties)


def unregister():
    async_loop_mgr = TexflowAsyncOperator.get_async_manager()
    async_loop_mgr.unregister()
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.texflow
    del bpy.app.driver_namespace["texflow_state"]
