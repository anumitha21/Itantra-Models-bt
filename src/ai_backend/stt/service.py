"""
STT Service Factory for instantiating STT Engines.
"""

from ai_backend.core.config import AppConfig, STTModelConfig
from ai_backend.core.exceptions import UnsupportedModel
from ai_backend.stt.base import BaseSTTEngine


class STTServiceFactory:
    """
    Factory for creating BaseSTTEngine instances based on model configuration.
    Uses lazy importing so missing non-selected dependencies do not crash app startup.
    """

    @staticmethod
    def create_engine(model_config: STTModelConfig, num_threads: int = 2) -> BaseSTTEngine:
        name_lower = model_config.name.lower()
        if name_lower == "indicconformer":
            from ai_backend.stt.indicconformer import IndicConformerSTTEngine
            return IndicConformerSTTEngine(config=model_config, num_threads=num_threads)
        elif name_lower in ["whisper", "whisper_tiny", "whisper_small"]:
            from ai_backend.stt.whisper_engine import WhisperSTTEngine
            return WhisperSTTEngine(config=model_config, num_threads=num_threads)
        elif name_lower == "mms":
            from ai_backend.stt.mms import MMSSTTEngine
            return MMSSTTEngine(config=model_config, num_threads=num_threads)
        raise UnsupportedModel(f"Unsupported STT model engine name '{model_config.name}'.")
