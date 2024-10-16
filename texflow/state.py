from dataclasses import dataclass
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


@dataclass
class TexflowState:
    pipe: DiffusionPipeline | None = None
    is_running: bool = False
    current_step: int = 0
