import torch
import inspect
import unittest
from PIL import Image

from texflow.utils import ROOT_PATH


def save_image(image: torch.Tensor, path):
    """
    image: h w c
    in range [0,1]
    """
    image = (image * 255).to(torch.uint8).numpy()
    Image.fromarray(image).save(path)


class TestCase(unittest.TestCase):
    def get_test_dir(self):
        class_name = self.__class__.__name__
        caller_name = inspect.currentframe().f_back.f_code.co_name
        dir = ROOT_PATH / "testoutput" / class_name / caller_name
        dir.mkdir(exist_ok=True, parents=True)
        return dir
