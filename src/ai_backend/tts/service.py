"""
TTS Service Placeholder.
"""

from ai_backend.core.exceptions import ServiceNotImplementedError
from ai_backend.core.types import AudioInput
from ai_backend.tts.base import BaseTTSEngine


class TTSService(BaseTTSEngine):
    """
    Placeholder service for future Text-to-Speech integration (e.g. Piper/VITS).
    """

    def load() -> None:
        raise ServiceNotImplementedError("TTS engine loading is not implemented in Stage 1.")

    def unload() -> None:
        pass

    def is_loaded(self) -> bool:
        return False

    def synthesize(self, text: str, language: str) -> AudioInput:
        raise ServiceNotImplementedError(
            "Text-to-Speech synthesis is not implemented in Stage 1."
        )
