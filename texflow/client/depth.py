import os
import bpy

from .utils import image_to_tensor


def get_render_pixels():
    for img in bpy.data.images:
        if img.type == "RENDER_RESULT":
            path = os.path.join(bpy.app.tempdir, "render_temp_save.exr")
            img.save_render(path)
            loaded_img = bpy.data.images.load(path)
            loaded_img.pixels[
                0
            ]  # this makes no sense, but it is necessary to load pixels array internally
            pixel_values = image_to_tensor(loaded_img)

            bpy.data.images.remove(loaded_img)
            try:
                os.remove(path)
            except:
                pass

            return pixel_values


def ensure_scene():
    render_scene_name = ".texflowRenderScene"
    if render_scene_name not in bpy.data.scenes:
        bpy.ops.scene.new(type="NEW")
        scene = bpy.data.scenes[-1]
        scene.name = render_scene_name
    else:
        scene = bpy.data.scenes[render_scene_name]

    bpy.context.window.scene = scene

    for object in bpy.context.scene.objects:
        object.select_set(True)

    bpy.ops.object.delete()

    return scene


def render_depth_map(
    obj: bpy.types.Object = None,
    camera_obj: bpy.types.Object | None = None,
    height=512,
    width=512,
    extra_background_distance=0.0,
):
    """
    obj: Object to be rendered
    camera: Camera to use for rendering depth, can be None in which case a
        camera with the same fov as the viewport will be used
    extra_background_distance: An extra distance to be added
      for the background behind the object
    """
    original_scene = bpy.context.scene
    temp_scene = ensure_scene()
    temp_scene.collection.objects.link(obj)

    if camera_obj is None:
        bpy.ops.object.camera_add()
        camera_obj = bpy.context.active_object
        screen_areas = bpy.context.screen.areas
        view_3d_areas = [a for a in screen_areas if a.type == "VIEW_3D"]
        if len(view_3d_areas) != 1:
            raise ValueError(
                f"Expecting a single view 3d area, but instead got {len(view_3d_areas)}!"
            )
        view_3d_space: bpy.types.SpaceView3D = view_3d_areas[0].spaces[0]
        camera_obj.data.lens = view_3d_space.lens
        bpy.ops.view3d.camera_to_view()

    else:
        assert camera_obj.type == "CAMERA"
        temp_scene.collection.objects.link(camera_obj)

    bpy.context.scene.camera = camera_obj

    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    view_layer = bpy.context.scene.view_layers[0]
    view_layer.use_pass_combined = False
    view_layer.use_pass_z = True

    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.image_settings.file_format = "OPEN_EXR"

    # Enable compositing nodes
    bpy.context.scene.use_nodes = True

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    nodes.clear()
    links.clear()

    render_layers_node = nodes.new("CompositorNodeRLayers")
    render_layers_node.location = (0, 0)

    output_node = nodes.new("CompositorNodeComposite")
    output_node.use_alpha = True
    output_node.location = (400, 0)

    links.new(render_layers_node.outputs["Depth"], output_node.inputs["Image"])
    links.new(render_layers_node.outputs["Alpha"], output_node.inputs["Alpha"])

    bpy.ops.render.render()

    depth_image = get_render_pixels()

    depth_image, occupancy = depth_image[..., 0], depth_image[..., 3]
    # I use a high threshold for the alpha
    # channel to esitmate the min/max of the depth image
    occupancy = occupancy > 0.99
    depth_values = depth_image[occupancy]
    # scales to 0,1
    min_value = depth_values.min()
    max_value = depth_values.max() + extra_background_distance
    scale = max_value - min_value
    scale = max(scale.item(), 1e-5)
    depth_image = (depth_image - min_value) / scale
    depth_image[~occupancy] = 1.0
    depth_image = depth_image.clamp_(0, 1)

    bpy.context.window.scene = original_scene
    bpy.data.scenes.remove(temp_scene)

    return depth_image, occupancy
