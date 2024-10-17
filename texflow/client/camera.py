from contextlib import contextmanager
import bpy

from .utils import select_obj


@contextmanager
def ensure_temp_camera(camera_obj: bpy.types.Object | None = None):
    """
    if camera_obj is not None, does nothing.
    But if the camera is none, returns a temporary camera that is
    aligned with the viewport. After exiting the context the temp
    camera is cleaned up.
    """
    added_camera = False
    old_camera = None
    if camera_obj is None:
        added_camera = True

        old_obj = bpy.context.active_object

        bpy.ops.object.mode_set(mode="OBJECT")

        old_camera = bpy.context.scene.camera
        bpy.ops.object.camera_add()
        camera_obj = bpy.context.active_object
        camera_obj.name = "texflow-camera"
        bpy.context.scene.camera = camera_obj

        select_obj(old_obj)
        bpy.ops.object.mode_set(mode="EDIT")
        screen_areas = bpy.context.screen.areas
        view_3d_areas = [a for a in screen_areas if a.type == "VIEW_3D"]
        if len(view_3d_areas) != 1:
            raise ValueError(
                f"Expecting a single view 3d area, but instead got {len(view_3d_areas)}!"
            )
        view_3d_space: bpy.types.SpaceView3D = view_3d_areas[0].spaces[0]
        camera_obj.data.lens = view_3d_space.lens
        bpy.ops.view3d.camera_to_view()

    assert camera_obj.type == "CAMERA"
    assert isinstance(camera_obj, bpy.types.Object)

    try:
        yield camera_obj
    finally:
        if added_camera:
            bpy.data.objects.remove(camera_obj)
            bpy.context.scene.camera = old_camera
