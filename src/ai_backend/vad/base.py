"""
Abstract VAD Engine Interface (Placeholder for Future Stage).
"""

from abc import ABC, abstractmethod
from typing import List
from ai_backend.core.types import AudioInput


class BaseVADEngine(ABC):
    """
    Abstract Base Class for Voice Activity Detection (VAD) Engines.
    """

    @abstractmethod
    def detect(self, audio: AudioInput) -> List[AudioInput]:
        """
        Detect speech segments in audio input and return list of speech AudioInput chunks.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset VAD internal state."""
        pass
