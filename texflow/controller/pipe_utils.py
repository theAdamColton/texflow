import inspect
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


def _post_pipe_init(pipe):
    pipe.safety_checker = None
    return pipe


def load_pipe(
    pretrained_model_or_path,
    controlnet_models_or_paths: list | None = None,
    token=None,
    dtype_override=None,
):
    kwargs = {}
    _, dtype = get_best_device_and_dtype()
    if dtype_override is not None:
        dtype = dtype_override
    kwargs["torch_dtype"] = dtype

    if controlnet_models_or_paths is not None:
        controlnets = [
            ControlNetModel.from_pretrained(path, torch_dtype=dtype)
            for path in controlnet_models_or_paths
        ]
        kwargs["controlnet"] = controlnets

    if token is not None:
        kwargs["token"] = token

    pipe = AutoPipelineForText2Image.from_pretrained(pretrained_model_or_path, **kwargs)
    pipe = _post_pipe_init(pipe)
    return pipe


def set_pipe_type(pipe, type="text2image"):
    if type == "text2image":
        pipe = AutoPipelineForText2Image.from_pipe(pipe)
    elif type == "image2image":
        pipe = AutoPipelineForImage2Image.from_pipe(pipe)
    elif type == "inpainting":
        pipe = AutoPipelineForInpainting.from_pipe(pipe)
    else:
        raise ValueError(type)
    pipe = _post_pipe_init(pipe)
    return pipe


def does_pipe_accept_negative_prompt(pipe):
    return "negative_prompt" in inspect.signature(pipe).parameters.keys()


def run_pipe(
    pipe,
    prompt=None,
    negative_prompt=None,
    image2image_image=None,
    image2image_strength: float = 0.8,
    inpainting_image=None,
    inpainting_mask_image=None,
    inpainting_strength=1.0,
    control_images: list | None = None,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 20,
    guidance_scale: float | None = 7.5,
    controlnet_conditioning_scales: None | list[float] = None,
    control_guidance_start: float = 0.0,
    control_guidance_end: float = 1.0,
    callback_on_step_end=None,
    seed: str | None = None,
):
    class_name = type(pipe).__name__
    is_img2img = "Img2Img" in class_name
    is_inpainting = "Inpaint" in class_name

    kwargs = dict(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        callback_on_step_end=callback_on_step_end,
    )

    if does_pipe_accept_negative_prompt(pipe):
        kwargs["negative_prompt"] = negative_prompt

    if hasattr(pipe, "controlnet") and pipe.controlnet is not None:
        controlnet_kwargs = dict(
            controlnet_conditioning_scale=controlnet_conditioning_scales,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
        )
        if is_img2img or is_inpainting:
            controlnet_kwargs["control_image"] = control_images
        else:
            controlnet_kwargs["image"] = control_images

        kwargs.update(controlnet_kwargs)

    if is_img2img:
        img2img_kwargs = dict(
            image=image2image_image,
            strength=image2image_strength,
        )
        kwargs.update(img2img_kwargs)

    if is_inpainting:
        inpainting_kwargs = dict(
            image=inpainting_image,
            strength=inpainting_strength,
            mask_image=inpainting_mask_image,
        )
        kwargs.update(inpainting_kwargs)

    if seed is not None:
        kwargs["generator"] = torch.Generator(pipe.device).manual_seed(int(seed))

    return pipe(**kwargs)
