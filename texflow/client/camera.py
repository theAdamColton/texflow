import bpy
import bmesh


def ensure_camera(camera_obj: bpy.types.Object | None = None):
    if camera_obj is None:
        bpy.ops.object.camera_add()
        camera_obj = bpy.context.active_object
        camera_obj.name = "texflow-camera"
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

    return camera_obj
