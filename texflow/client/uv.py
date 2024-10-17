import bpy_extras
import bpy
import bmesh

from .utils import select_obj
from .scene import ensure_texflow_temp_scene


def uv_proj(
    obj: bpy.types.Object,
    camera_obj: bpy.types.Object = None,
    height=512,
    width=512,
):
    mesh = obj.data
    assert isinstance(mesh, bpy.types.Mesh)
    assert bpy.context.mode == "EDIT_MESH"

    bpy.ops.mesh.uv_texture_add()
    new_uv_layer = mesh.uv_layers[-1]

    with ensure_texflow_temp_scene() as temp_scene:
        temp_scene.collection.objects.link(obj)
        temp_scene.collection.objects.link(camera_obj)
        temp_scene.render.resolution_y = height
        temp_scene.render.resolution_x = width

        select_obj(obj)

        bm = bmesh.from_edit_mesh(mesh)
        bm_new_uv_layer = bm.loops.layers.uv.verify()
        for face in bm.faces:
            for loop in face.loops:
                loop_uv = loop[bm_new_uv_layer]
                vert = loop.vert
                if not vert.select:
                    continue

                co_2d = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, camera_obj, vert.co
                )

                loop_uv.uv[0] = co_2d.x
                loop_uv.uv[1] = co_2d.y

        bmesh.update_edit_mesh(mesh)

    return new_uv_layer
