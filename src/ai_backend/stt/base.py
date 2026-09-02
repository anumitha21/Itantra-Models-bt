"""
Abstract STT Engine Interface for AI Backend Foundation.
"""

from abc import ABC, abstractmethod
from ai_backend.core.types import AudioInput, TranscriptionResult
from ai_backend.models.model_metadata import ModelMetadata


class BaseSTTEngine(ABC):
    """
    Abstract Base Class for Speech-to-Text Engines.
    Decouples core application logic from specific STT framework or ONNX implementations.
    """

    @abstractmethod
    def load(self) -> None:
        """Initialize and load model weights into memory."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model weights and release memory resources."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the engine model is currently loaded in memory."""
        pass

    @abstractmethod
    def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        """
        Transcribe audio input and return structured TranscriptionResult object.
        """
        pass

    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Return model metadata."""
        pass
