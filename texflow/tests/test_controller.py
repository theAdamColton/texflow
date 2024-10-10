import inspect
from PIL import Image
import torch
import unittest

from ..controller.pipe_utils import load_pipe, run_pipe, set_pipe_type


class TestController(unittest.TestCase):
    def test_load_pipe_stable_diffusion(self):
        pipe = load_pipe(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            dtype_override=torch.float32,
        )
        output = pipe(prompt="a prompt about dogs", num_inference_steps=2)
        image = output["images"][0]
        self.assertIsInstance(image, Image.Image)

    def assertInSignature(self, key, function):
        self.assertIn(key, inspect.signature(function).parameters.keys())

    def assertNotInSignature(self, key, function):
        self.assertNotIn(key, inspect.signature(function).parameters.keys())

    def test_set_pipe_type(self):
        pipe = load_pipe(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
        )
        pipe = set_pipe_type(pipe, type="text2image")
        self.assertNotInSignature("image", pipe)
        pipe = set_pipe_type(pipe, type="image2image")
        self.assertInSignature("image", pipe)
        pipe = set_pipe_type(pipe, type="inpainting")
        self.assertInSignature("image", pipe)

    def test_load_controlnet(self):
        pipe = load_pipe(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            controlnet_models_or_paths=["hf-internal-testing/tiny-controlnet"],
        )
        self.assertInSignature("image", pipe)
        pipe = set_pipe_type(pipe, "image2image")
        self.assertInSignature("image", pipe)
        self.assertInSignature("control_image", pipe)

    def test_run_pipe_stable_diffusion_text2image(self):
        pipe = load_pipe(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            dtype_override=torch.float32,
        )
        output = run_pipe(
            pipe, prompt="rabbit", num_inference_steps=2, height=64, width=64
        )
        image = output["images"][0]
        self.assertIsInstance(image, Image.Image)

    def test_run_pipe_stable_diffusion_image2image(self):
        pipe = load_pipe(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            dtype_override=torch.float32,
        )
        pipe = set_pipe_type(pipe, "image2image")
        image = (torch.rand(3, 64, 64) * 255).to(torch.uint8)
        output = run_pipe(
            pipe,
            prompt="rabbit",
            image2image_image=image,
            num_inference_steps=2,
            height=64,
            width=64,
        )
        image = output["images"][0]
        self.assertIsInstance(image, Image.Image)

    def test_run_pipe_stable_diffusion_inpaint(self):
        pipe = load_pipe(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            dtype_override=torch.float32,
        )
        pipe = set_pipe_type(pipe, "inpainting")
        image = (torch.rand(3, 64, 64) * 255).to(torch.uint8)
        mask_image = (torch.randn(64, 64) > 0) * 1.0
        output = run_pipe(
            pipe,
            prompt="rabbit",
            inpainting_image=image,
            inpainting_mask_image=mask_image,
            num_inference_steps=2,
            height=64,
            width=64,
        )
        image = output["images"][0]
        self.assertIsInstance(image, Image.Image)
