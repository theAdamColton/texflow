import torch
import unittest

from ..controller.pipe_utils import load_pipe, set_pipe_type


class TestController(unittest.TestCase):
    def test_load_pipe_stable_diffusion(self):
        pipe = load_pipe(
            "hf-internal-testing/tiny-stable-diffusion-pipe",
            force_cpu=True,
            dtype_override=torch.float32,
        )
