import torch
from diffusers import ControlNetModel
from diffusers.pipelines.auto_pipeline import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
)


def get_best_device_and_dtype():
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")

        # Check if GPU supports fp16
        if torch.cuda.get_device_capability()[0] >= 7:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

    # If CUDA is not available, check for MPS (Apple Silicon)
    elif torch.backends.mps.is_available():
        # mps doesn't have full support
        device = torch.device("cpu")
        dtype = torch.float32

    # If neither CUDA nor MPS is available, use CPU
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    return device, dtype


def load_pipe(
    pretrained_model_or_path,
    controlnet_models_or_paths: list | None = None,
    token=None,
    force_cpu=False,
    dtype_override=None,
):
    _, dtype = get_best_device_and_dtype()
    if dtype_override is not None:
        dtype = dtype_override

    device_map = None
    if force_cpu:
        device_map = "cpu"

    if controlnet_models_or_paths is not None:
        controlnets = [
            ControlNetModel(path, torch_dtype=dtype, device_map=device_map)
            for path in controlnet_models_or_paths
        ]
    else:
        controlnets = None
    pipe = AutoPipelineForText2Image.from_pretrained(
        pretrained_model_or_path,
        controlnet=controlnets,
        torch_dtype=dtype,
        token=token,
        device_map=device_map,
    )
    return pipe


def set_pipe_type(pipe, type="text2image"):
    if type == "text2image":
        return AutoPipelineForText2Image.from_pipe(pipe)
    elif type == "image2image":
        return AutoPipelineForImage2Image.from_pipe(pipe)
    elif type == "inpainting":
        return AutoPipelineForInpainting.from_pipe(pipe)
