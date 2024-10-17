import bpy
from contextlib import contextmanager


@contextmanager
def ensure_texflow_temp_scene():
    original_scene = bpy.context.scene

    render_scene_name = ".texflowRenderScene"
    if render_scene_name not in bpy.data.scenes:
        bpy.ops.scene.new(type="NEW")
        temp_scene = bpy.data.scenes[-1]
        temp_scene.name = render_scene_name
    else:
        temp_scene = bpy.data.scenes[render_scene_name]

    bpy.context.window.scene = temp_scene

    for object in bpy.context.scene.objects:
        object.select_set(True)

    bpy.ops.object.delete()

    try:
        yield temp_scene
    finally:
        bpy.context.window.scene = original_scene
        bpy.data.scenes.remove(temp_scene)
