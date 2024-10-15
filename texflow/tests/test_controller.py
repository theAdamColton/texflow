import numpy as np
import inspect
import torch
import unittest

from ..controller.pipe_utils import load_pipe, run_pipe, set_pipe_type


class TestPipe(unittest.TestCase):
    def _check_image(self, image):
        h, w, c = image.shape
        self.assertIsInstance(image, np.ndarray)

    def test_load_pipe_stable_diffusion(self):
        pipe = load_pipe(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            dtype_override=torch.float32,
        )
        image = pipe(
            prompt="a prompt about dogs", num_inference_steps=2, output_type="np"
        )["images"][0]
        self._check_image(image)

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
        self._test_text2image("hf-internal-testing/tiny-stable-diffusion-pipe")

    def test_run_pipe_stable_diffusion_xl_text2image(self):
        self._test_text2image("hf-internal-testing/tiny-stable-diffusion-xl-pipe")

    def test_run_pipe_stable_diffusion_xl_image2image(self):
        self._test_image2image("hf-internal-testing/tiny-stable-diffusion-xl-pipe")

    def test_run_pipe_stable_diffusion_xl_controlnet_text2image(self):
        self._test_controlnet_text2image(
            "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
            "hf-internal-testing/tiny-controlnet-sdxl",
        )

    def test_run_pipe_flux_text2image(self):
        self._test_text2image(
            "hf-internal-testing/tiny-flux-pipe",
        )

    def test_run_pipe_stable_diffusion_image2image(self):
        self._test_image2image("hf-internal-testing/tiny-stable-diffusion-pipe")

    def test_run_pipe_stable_diffusion_inpaint(self):
        self._test_inpainting("hf-internal-testing/tiny-stable-diffusion-pipe")

    def test_run_pipe_stable_diffusion_controlnet_text2image(self):
        self._test_controlnet_text2image(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            "hf-internal-testing/tiny-controlnet",
        )

    def test_run_pipe_stable_diffusion_controlnet_image2image(self):
        self._test_controlnet_image2image(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            "hf-internal-testing/tiny-controlnet",
        )

    def test_run_pipe_stable_diffusion_controlnet_inpainting(self):
        self._test_controlnet_inpainting(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            "hf-internal-testing/tiny-controlnet",
        )

    def _test_controlnet_inpainting(self, base_name, controlnet_name):
        pipe = load_pipe(
            base_name,
            controlnet_models_or_paths=[controlnet_name],
        )
        pipe = set_pipe_type(pipe, "inpainting")
        controlnet_images = [torch.rand(1, 3, 64, 64)]
        inpainting_image = torch.rand(3, 64, 64)
        inpainting_mask = torch.zeros(64, 64)
        output = run_pipe(
            pipe,
            prompt="rabbit",
            control_images=controlnet_images,
            inpainting_image=inpainting_image,
            inpainting_mask_image=inpainting_mask,
            controlnet_conditioning_scales=[0.5],
            num_inference_steps=2,
            height=64,
            width=64,
        )
        self._check_image(output)

    def _test_controlnet_image2image(self, base_name, controlnet_name):
        pipe = load_pipe(
            base_name,
            controlnet_models_or_paths=[controlnet_name],
        )
        pipe = set_pipe_type(pipe, "image2image")
        controlnet_images = [torch.rand(1, 3, 64, 64)]
        image2image_image = torch.rand(3, 64, 64)
        output = run_pipe(
            pipe,
            prompt="rabbit",
            control_images=controlnet_images,
            image2image_image=image2image_image,
            controlnet_conditioning_scales=[0.5],
            num_inference_steps=2,
            height=64,
            width=64,
        )
        self._check_image(output)

    def _test_controlnet_text2image(self, base_name, controlnet_name):
        pipe = load_pipe(
            base_name,
            controlnet_models_or_paths=[controlnet_name],
        )
        controlnet_images = [torch.rand(1, 3, 64, 64)]
        output = run_pipe(
            pipe,
            prompt="rabbit",
            control_images=controlnet_images,
            controlnet_conditioning_scales=[0.5],
            num_inference_steps=2,
            height=64,
            width=64,
        )
        self._check_image(output)

    def _test_inpainting(self, name):
        pipe = load_pipe(
            name,
            dtype_override=torch.float32,
        )
        pipe = set_pipe_type(pipe, "inpainting")
        image = (torch.rand(3, 64, 64) * 255).to(torch.uint8)
        mask_image = (torch.rand(64, 64) > 0) * 1.0
        output = run_pipe(
            pipe,
            prompt="rabbit",
            inpainting_image=image,
            inpainting_mask_image=mask_image,
            num_inference_steps=2,
            height=64,
            width=64,
        )
        self._check_image(output)

    def _test_image2image(self, name):
        pipe = load_pipe(
            name,
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
        self._check_image(output)

    def _test_text2image(self, name):
        pipe = load_pipe(
            name,
            dtype_override=torch.float32,
        )
        output = run_pipe(
            pipe, prompt="rabbit", num_inference_steps=2, height=64, width=64
        )
        self._check_image(output)
