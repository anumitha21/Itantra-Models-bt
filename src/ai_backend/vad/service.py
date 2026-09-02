"""
VAD Service Placeholder.
"""

from typing import List
from ai_backend.core.exceptions import ServiceNotImplementedError
from ai_backend.core.types import AudioInput
from ai_backend.vad.base import BaseVADEngine


class VADService(BaseVADEngine):
    """
    Placeholder service for future Voice Activity Detection (e.g. Silero VAD / Ten VAD).
    """

    def detect(self, audio: AudioInput) -> List[AudioInput]:
        raise ServiceNotImplementedError(
            "Voice Activity Detection is not implemented in Stage 1."
        )

    def reset(self) -> None:
        pass
