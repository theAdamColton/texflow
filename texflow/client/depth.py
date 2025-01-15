import numpy as np
import os
import bpy

from .scene import ensure_texflow_temp_scene
from .utils import image_to_arr


def get_render_pixels():
    for img in bpy.data.images:
        if img.type == "RENDER_RESULT":
            path = os.path.join(bpy.app.tempdir, "render_temp_save.exr")
            img.save_render(path)
            loaded_img = bpy.data.images.load(path)
            loaded_img.pixels[
                0
            ]  # this makes no sense, but it is necessary to load pixels array internally
            pixel_values = image_to_arr(loaded_img)

            bpy.data.images.remove(loaded_img)
            try:
                os.remove(path)
            except:
                pass

            return pixel_values
    raise ValueError("No render result image found")


def render_depth_map(
    obj: bpy.types.Object = None,
    camera_obj: bpy.types.Object = None,
    extra_background_distance=0.0,
):
    """
    obj: Object to be rendered
    camera: Camera to use for rendering depth, can be None in which case a
        camera with the same fov as the viewport will be used
    extra_background_distance: Extra distance to be added
      for the background behind the object
    """

    # create a new scene to ensure that
    # render settings are all as expected
    with ensure_texflow_temp_scene() as temp_scene:
        temp_scene.collection.objects.link(obj)
        temp_scene.collection.objects.link(camera_obj)
        temp_scene.camera = camera_obj

        temp_scene.render.engine = "BLENDER_EEVEE_NEXT"
        view_layer = temp_scene.view_layers[0]
        view_layer.use_pass_combined = False
        view_layer.use_pass_z = True

        temp_scene.render.film_transparent = True
        temp_scene.render.image_settings.file_format = "OPEN_EXR"

        # Enable compositing nodes
        temp_scene.use_nodes = True

        nodes = temp_scene.node_tree.nodes
        links = temp_scene.node_tree.links
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
        depth_image = np.clip(depth_image, 0, 1)

    return depth_image, occupancy
