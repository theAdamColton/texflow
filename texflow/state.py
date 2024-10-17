from dataclasses import dataclass
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


@dataclass
class TexflowState:
    pipe: DiffusionPipeline | None = None
    status: str = "PENDING"  # or "RUNNING" or "LOADING"
    current_step: int = 0

    load_step: int = 0
    load_max_step: int = 0
