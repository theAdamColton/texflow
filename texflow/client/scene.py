import bpy
from contextlib import contextmanager


@contextmanager
def ensure_texflow_temp_scene(height: int | None = None, width: int | None = None):
    original_scene = bpy.context.scene

    # Defaults to the render resolution of the current scene
    if height is None:
        height = original_scene.render.resolution_y
    if width is None:
        width = original_scene.render.resolution_x

    render_scene_name = ".texflowRenderScene"
    if render_scene_name not in bpy.data.scenes:
        bpy.ops.scene.new(type="NEW")
        temp_scene = bpy.data.scenes[-1]
        temp_scene.name = render_scene_name
    else:
        temp_scene = bpy.data.scenes[render_scene_name]

    temp_scene.render.resolution_y = height
    temp_scene.render.resolution_x = width

    bpy.context.window.scene = temp_scene

    for object in bpy.context.scene.objects:
        object.select_set(True)

    bpy.ops.object.delete()

    try:
        yield temp_scene
    finally:
        bpy.context.window.scene = original_scene
        bpy.data.scenes.remove(temp_scene)
