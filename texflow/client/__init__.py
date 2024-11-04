import gc
import asyncio
import bpy
import logging

from ..state import TexflowState
from ..utils import DESCRIPTION, VERSION_TUPLE
from .ui import TexflowProperties, TexflowPanel, RenderDepthImageOperator
from .async_loop import AsyncLoopModalOperator, AsyncModalOperatorMixin

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
    AsyncLoopModalOperator,
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
