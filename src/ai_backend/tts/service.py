"""
TTS Service Factory for instantiating Text-to-Speech Engines.
"""

from ai_backend.core.config import TTSModelConfig
from ai_backend.core.exceptions import UnsupportedModel
from ai_backend.tts.base import BaseTTSEngine


class TTSServiceFactory:
    """
    Factory for creating BaseTTSEngine instances based on model configuration.
    Uses lazy importing so missing non-selected dependencies do not crash app startup.
    """

    @staticmethod
    def create_engine(model_config: TTSModelConfig, num_threads: int = 2) -> BaseTTSEngine:
        name_lower = model_config.name.lower()
        if name_lower in ["ai4bharat_vits", "mms_vits", "vits", "indic_tts", "mms"]:
            from ai_backend.tts.vits_engine import VitsTTSEngine
            return VitsTTSEngine(config=model_config, num_threads=num_threads)
        raise UnsupportedModel(f"Unsupported TTS model engine name '{model_config.name}'.")


TTSService = TTSServiceFactory
