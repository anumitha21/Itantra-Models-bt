"""
Future-ready SpeechPipeline Orchestrator.
Combines ModelManager, STT, TTS (stub), and VAD (stub).
"""

from typing import Optional
from ai_backend.core.config import AppConfig
from ai_backend.core.types import AudioInput, TranscriptionResult
from ai_backend.models.model_manager import ModelManager
from ai_backend.tts.service import TTSService
from ai_backend.vad.service import VADService


class SpeechPipeline:
    """
    Unified Pipeline Interface for offline speech operations.
    """

    def __init__(
        self,
        app_config: Optional[AppConfig] = None,
        model_manager: Optional[ModelManager] = None
    ):
        self.app_config = app_config or AppConfig.load()
        self.model_manager = model_manager or ModelManager(self.app_config)
        self.tts_service = TTSService()
        self.vad_service = VADService()

    def transcribe(
        self,
        audio: AudioInput,
        language: str = "hi",
        model_name: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe input audio to text using the configured STT engine for the specified language.
        """
        stt_engine = self.model_manager.get_stt(language, model_name=model_name)
        return stt_engine.transcribe(audio)

    def synthesize(self, text: str, language: str = "hi") -> AudioInput:
        """
        Synthesize input text to audio (placeholder for future TTS integration).
        """
        return self.tts_service.synthesize(text, language)

    def process(
        self,
        audio: AudioInput,
        language: str = "hi",
        model_name: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Complete speech pipeline processing (VAD [stub] -> STT -> return result).
        """
        return self.transcribe(audio, language=language, model_name=model_name)
