"""
Memory-aware ModelManager for discovering, loading, and unloading AI models.
"""

from typing import Dict, Optional
from ai_backend.core.config import AppConfig
from ai_backend.core.logging import get_logger
from ai_backend.models.model_registry import ModelRegistry
from ai_backend.stt.base import BaseSTTEngine
from ai_backend.stt.service import STTServiceFactory
from ai_backend.tts.base import BaseTTSEngine
from ai_backend.tts.service import TTSServiceFactory

logger = get_logger("ModelManager")


class ModelManager:
    """
    Manages model loading/unloading lifecycle for STT and TTS engines.
    Ensures single-model RAM discipline by default (unloading previous models)
    to satisfy strict 2-3 GB RAM Android / field edge device targets.
    Benchmark multi-model caching is strictly opt-in via benchmark_mode=True.
    """

    def __init__(self, app_config: Optional[AppConfig] = None, benchmark_mode: bool = False):
        self.app_config = app_config or AppConfig.load()
        self.registry = ModelRegistry(self.app_config)
        self.benchmark_mode = benchmark_mode
        self._stt_engines: Dict[str, BaseSTTEngine] = {}
        self._tts_engines: Dict[str, BaseTTSEngine] = {}
        self._active_stt_key: Optional[str] = None
        self._active_stt_lang: Optional[str] = None
        self._active_tts_key: Optional[str] = None
        self._active_tts_lang: Optional[str] = None

    def _get_model_key(self, language: str, model_name: Optional[str] = None) -> str:
        lang_code = self.registry.normalize_language(language)
        if model_name:
            return f"{model_name.lower().strip()}_{lang_code}"
        return lang_code

    # -------------------------------------------------------------
    # STT Model Lifecycle
    # -------------------------------------------------------------
    def load_stt(self, language: str, model_name: Optional[str] = None) -> BaseSTTEngine:
        """
        Load STT model for the specified language and optional model name.
        Under default mode (benchmark_mode=False), unloads any previously loaded model to save RAM.
        """
        lang_code = self.registry.normalize_language(language)
        model_key = self._get_model_key(language, model_name)

        # Single-model RAM discipline by default
        if not self.benchmark_mode:
            if self._active_stt_key and self._active_stt_key != model_key:
                logger.info(f"Unloading active STT model [{self._active_stt_key}] to free memory for [{model_key}].")
                self.unload_stt_key(self._active_stt_key)

        if model_key not in self._stt_engines:
            stt_cfg = self.registry.get_stt_config(lang_code, model_name=model_name)
            engine = STTServiceFactory.create_engine(
                model_config=stt_cfg,
                num_threads=self.app_config.num_threads
            )
            self._stt_engines[model_key] = engine

        engine = self._stt_engines[model_key]
        if not engine.is_loaded():
            engine.load()

        self._active_stt_key = model_key
        self._active_stt_lang = lang_code
        return engine

    def get_stt(self, language: str, model_name: Optional[str] = None) -> BaseSTTEngine:
        """
        Retrieve loaded STT engine for specified language / model. Automatically loads if not loaded.
        """
        model_key = self._get_model_key(language, model_name)
        if model_key not in self._stt_engines or not self._stt_engines[model_key].is_loaded():
            return self.load_stt(language, model_name=model_name)
        return self._stt_engines[model_key]

    def unload_stt_key(self, model_key: str) -> None:
        """
        Unload specific STT engine by cache key.
        """
        if model_key in self._stt_engines:
            self._stt_engines[model_key].unload()
            del self._stt_engines[model_key]
        if self._active_stt_key == model_key:
            self._active_stt_key = None
            self._active_stt_lang = None

    def unload_stt(self, language: str) -> None:
        """
        Unload STT engine for specified language from memory.
        """
        lang_code = self.registry.normalize_language(language)
        keys_to_unload = [k for k in self._stt_engines.keys() if k == lang_code or k.endswith(f"_{lang_code}")]
        for k in keys_to_unload:
            self.unload_stt_key(k)

    # -------------------------------------------------------------
    # TTS Model Lifecycle
    # -------------------------------------------------------------
    def load_tts(self, language: str, model_name: Optional[str] = None) -> BaseTTSEngine:
        """
        Load TTS model for the specified language and optional model name.
        Under default mode (benchmark_mode=False), unloads any previously loaded TTS model.
        """
        lang_code = self.registry.normalize_language(language)
        model_key = self._get_model_key(language, model_name)

        if not self.benchmark_mode:
            if self._active_tts_key and self._active_tts_key != model_key:
                logger.info(f"Unloading active TTS model [{self._active_tts_key}] to free memory for [{model_key}].")
                self.unload_tts_key(self._active_tts_key)

        if model_key not in self._tts_engines:
            tts_cfg = self.registry.get_tts_config(lang_code, model_name=model_name)
            engine = TTSServiceFactory.create_engine(
                model_config=tts_cfg,
                num_threads=self.app_config.num_threads
            )
            self._tts_engines[model_key] = engine

        engine = self._tts_engines[model_key]
        if not engine.is_loaded():
            engine.load()

        self._active_tts_key = model_key
        self._active_tts_lang = lang_code
        return engine

    def get_tts(self, language: str, model_name: Optional[str] = None) -> BaseTTSEngine:
        """
        Retrieve loaded TTS engine for specified language / model. Automatically loads if not loaded.
        """
        model_key = self._get_model_key(language, model_name)
        if model_key not in self._tts_engines or not self._tts_engines[model_key].is_loaded():
            return self.load_tts(language, model_name=model_name)
        return self._tts_engines[model_key]

    def unload_tts_key(self, model_key: str) -> None:
        """
        Unload specific TTS engine by cache key.
        """
        if model_key in self._tts_engines:
            self._tts_engines[model_key].unload()
            del self._tts_engines[model_key]
        if self._active_tts_key == model_key:
            self._active_tts_key = None
            self._active_tts_lang = None

    def unload_tts(self, language: str) -> None:
        """
        Unload TTS engine for specified language from memory.
        """
        lang_code = self.registry.normalize_language(language)
        keys_to_unload = [k for k in self._tts_engines.keys() if k == lang_code or k.endswith(f"_{lang_code}")]
        for k in keys_to_unload:
            self.unload_tts_key(k)

    def unload_all(self) -> None:
        """
        Unload all loaded models (STT & TTS).
        """
        for model_key in list(self._stt_engines.keys()):
            self.unload_stt_key(model_key)
        for model_key in list(self._tts_engines.keys()):
            self.unload_tts_key(model_key)
        self._active_stt_key = None
        self._active_stt_lang = None
        self._active_tts_key = None
        self._active_tts_lang = None
