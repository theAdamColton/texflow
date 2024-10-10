import bpy

from ..state import TexflowState
from ..utils import DESCRIPTION, VERSION_TUPLE


bl_info = {
    "name": "texflow",
    "author": "Adam Colton",
    "description": DESCRIPTION,
    "blender": (4, 2, 1),
    "version": VERSION_TUPLE,
    "location": "Image Editor -> Sidebar -> texflow",
    "category": "Paint",
}

classes = ()


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.app.driver_namespace["texflow_state"] = TexflowState()


def unregister():
    for cls in classes:
        bpy.utils.register_class(cls)
    del bpy.app.driver_namespace["texflow_state"]
