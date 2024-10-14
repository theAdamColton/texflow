import numpy as np
import torch
import bpy


def image_to_tensor(image: bpy.types.Image):
    width = image.size[0]
    height = image.size[1]
    channels = image.channels

    arr = np.array(image.pixels[:], dtype=np.float32)
    arr = arr.reshape((height, width, channels))
    return torch.from_numpy(arr)


def select_obj(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
