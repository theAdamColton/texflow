from PIL import Image
import torch
import unittest

from ..controller.pipe_utils import load_pipe, set_pipe_type


class TestController(unittest.TestCase):
    def test_load_pipe_stable_diffusion(self):
        pipe = load_pipe(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            dtype_override=torch.float32,
        )
        output = pipe(prompt="a prompt about dogs", num_inference_steps=2)
        image = output["images"][0]
        self.assertIsInstance(image, Image.Image)
