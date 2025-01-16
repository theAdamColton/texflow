from dataclasses import dataclass
import uuid
from enum import Enum


@dataclass
class TexflowState:
    """
    Transient state that is not saved as part of .blend files
    """

    render_id: str | None = None
    active_obj = None
