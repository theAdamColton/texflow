import bpy
import bmesh

from .camera import ensure_camera


def uv_proj(
    obj: bpy.types.Object,
    camera_obj: bpy.types.Object | None = None,
    height=512,
    width=512,
):
    mesh = obj.data
    assert isinstance(mesh, bpy.types.Mesh)

    camera_obj = ensure_camera(camera_obj)
    bm = bmesh.from_edit_mesh(mesh)

    bpy.ops.mesh.uv_texture_add()
    new_uv_layer = mesh.uv_layers[-1]
    mesh.uv_layers.active = new_uv_layer

    # uv unwrap using camera view
    camera_data: bpy.types.Camera = camera_obj.data
    camera_matrix = camera_obj.matrix_world.inverted()
    projection_matrix = camera_obj.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(),
        x=width,
        y=height,
        scale_x=1,
        scale_y=1,
    )

    bm_new_uv_layer = bm.loops.layers.uv.verify()
    for face in bm.faces:
        for loop in face.loops:
            loop_uv = loop[bm_new_uv_layer]
            vert = loop.vert
            if not vert.select:
                continue
            co_3d = obj.matrix_world @ loop.vert.co
            co_2d = projection_matrix @ camera_matrix @ co_3d
            if co_2d.z < 0:  # Check if the point is behind the camera
                continue
            loop_uv.uv[0] = co_2d.x / co_2d.z / (height / width) + 0.5
            loop_uv.uv[1] = co_2d.y / co_2d.z + 0.5
    bmesh.update_edit_mesh(mesh)

    return new_uv_layer
