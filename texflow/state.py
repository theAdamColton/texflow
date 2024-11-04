from dataclasses import dataclass
import uuid
from enum import Enum


class TexflowStatus(Enum):
    """
    pending: Not currently connected to comfyui
    connecting: Trying to initiate a connection to the comfyui server
    ready: Connected and ready to recieve requests to render depth images
    running: Currently rendering and uploading a depth image to the comfyui server
    """

    PENDING = 1
    CONNECTING = 2
    READY = 3
    RUNNING = 4


@dataclass
class TexflowState:
    """
    Transient state that is not saved as part of .blend files
    """

    status: TexflowStatus = TexflowStatus.PENDING
    client_id = uuid.uuid4()
