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
    def load(self) -> None:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        pass

    @abstractmethod
    def synthesize(self, text: str, language: str) -> AudioInput:
        """
        Synthesize input text string to AudioInput.
        """
        pass
