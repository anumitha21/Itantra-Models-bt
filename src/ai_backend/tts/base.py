"""
Abstract TTS Engine Interface (Placeholder for Future Stage).
"""

from abc import ABC, abstractmethod
from ai_backend.core.types import AudioInput


class BaseTTSEngine(ABC):
    """
    Abstract Base Class for Text-to-Speech (TTS) Engines.
    """

    @abstractmethod
    def load() -> None:
        pass

    @abstractmethod
    def unload() -> None:
        pass

    @abstractmethod
    def is_loaded() -> bool:
        pass

    @abstractmethod
    def synthesize(self, text: str, language: str) -> AudioInput:
        """
        Synthesize input text string to AudioInput.
        """
        pass
