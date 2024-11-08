import numpy as np
from PIL import Image
import bpy


def image_to_arr(image: bpy.types.Image):
    width = image.size[0]
    height = image.size[1]
    channels = image.channels

    arr = np.array(image.pixels[:], dtype=np.float16)
    arr = arr.reshape((height, width, channels))
    return arr


def select_obj(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)


def to_image16(arr: np.ndarray):
    arr = np.clip(arr, 0, 1)
    arr = (arr * 2**16).astype(np.uint16)
    return Image.fromarray(arr, "I;16")
