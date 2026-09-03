"""
Model Registry for discovering and configuring available AI models.
"""

from typing import Dict, List, Optional
from ai_backend.core.config import AppConfig, STTModelConfig, TTSModelConfig
from ai_backend.core.exceptions import UnsupportedLanguage, UnsupportedModel
from ai_backend.models.model_metadata import ModelMetadata


# Language aliases
LANG_ALIASES: Dict[str, str] = {
    "hindi": "hi",
    "hi": "hi",
    "hin": "hi",
    "english": "en",
    "en": "en",
    "eng": "en",
    "tamil": "ta",
    "ta": "ta",
    "tam": "ta",
    "telugu": "te",
    "te": "te",
    "tel": "te",
}


class ModelRegistry:
    """
    Registry maintaining metadata and file path mapping for AI models (STT & TTS).
    """

    def __init__(self, app_config: Optional[AppConfig] = None):
        self.app_config = app_config or AppConfig.load()
        self._stt_configs = self.app_config.stt_models
        self._tts_configs = self.app_config.tts_models

    def normalize_language(self, language: str) -> str:
        lang_key = language.lower().strip()
        if lang_key in LANG_ALIASES:
            return LANG_ALIASES[lang_key]
        if lang_key in self._stt_configs or lang_key in self._tts_configs:
            return lang_key
        # Check if language matches language field of any config
        for cfg in list(self._stt_configs.values()) + list(self._tts_configs.values()):
            if cfg.language == lang_key:
                return lang_key
        supported = list(set(list(LANG_ALIASES.keys()) + [cfg.language for cfg in list(self._stt_configs.values()) + list(self._tts_configs.values())]))
        raise UnsupportedLanguage(
            f"Unsupported language '{language}'. Supported options: {supported}"
        )

    def get_stt_config(self, language: str, model_name: Optional[str] = None) -> STTModelConfig:
        lang_code = self.normalize_language(language)

        if model_name is None:
            if lang_code in self._stt_configs:
                return self._stt_configs[lang_code]
            for cfg in self._stt_configs.values():
                if cfg.language == lang_code:
                    return cfg
            raise UnsupportedModel(f"No STT model registered for language code '{lang_code}'.")

        m_name = model_name.lower().strip()
        lookup_key = f"{m_name}_{lang_code}"

        if lookup_key in self._stt_configs:
            return self._stt_configs[lookup_key]

        if m_name in ["indicconformer", "indic_conformer"] and lang_code in self._stt_configs:
            return self._stt_configs[lang_code]

        # Scan configs for matching model name / model size and language
        for key, cfg in self._stt_configs.items():
            if cfg.language == lang_code:
                if cfg.name.lower() == m_name or key.startswith(m_name) or getattr(cfg, "model_size", None) == m_name:
                    return cfg

        raise UnsupportedModel(f"No STT model registered for engine '{model_name}' and language '{lang_code}'.")

    def get_stt_metadata(self, language: str, model_name: Optional[str] = None) -> ModelMetadata:
        cfg = self.get_stt_config(language, model_name=model_name)
        return ModelMetadata(
            name=cfg.name,
            version=cfg.version,
            language=cfg.language,
            format=cfg.format,
            quantization=cfg.quantization,
            expected_sample_rate=cfg.expected_sample_rate,
            architecture=cfg.architecture,
            source=cfg.source,
            runtime=cfg.runtime,
            extra={"model_size": cfg.model_size, "device": cfg.device},
        )

    def list_supported_stt_languages(self) -> List[str]:
        langs = set()
        for cfg in self._stt_configs.values():
            langs.add(cfg.language)
        return sorted(list(langs))

    def list_all_stt_configs(self) -> Dict[str, STTModelConfig]:
        return self._stt_configs

    def get_tts_config(self, language: str, model_name: Optional[str] = None) -> TTSModelConfig:
        lang_code = self.normalize_language(language)

        if model_name is None:
            for key, cfg in self._tts_configs.items():
                if cfg.language == lang_code:
                    return cfg
            raise UnsupportedModel(f"No TTS model registered for language code '{lang_code}'.")

        m_name = model_name.lower().strip()
        lookup_key = f"{m_name}_{lang_code}"

        if lookup_key in self._tts_configs:
            return self._tts_configs[lookup_key]

        # Search by prefix or model name
        for key, cfg in self._tts_configs.items():
            if cfg.language == lang_code:
                if cfg.name.lower() == m_name or key.startswith(m_name) or m_name in cfg.name.lower():
                    return cfg

        raise UnsupportedModel(f"No TTS model registered for engine '{model_name}' and language '{lang_code}'.")

    def get_tts_metadata(self, language: str, model_name: Optional[str] = None) -> ModelMetadata:
        cfg = self.get_tts_config(language, model_name=model_name)
        return ModelMetadata(
            name=cfg.name,
            version=cfg.version,
            language=cfg.language,
            format=cfg.format,
            quantization=cfg.quantization,
            expected_sample_rate=cfg.expected_sample_rate,
            architecture=cfg.architecture,
            source=cfg.source,
            runtime=cfg.runtime,
            extra={"speaker_id": cfg.speaker_id, "device": cfg.device},
        )

    def list_supported_tts_languages(self) -> List[str]:
        langs = set()
        for cfg in self._tts_configs.values():
            langs.add(cfg.language)
        return sorted(list(langs))

    def list_all_tts_configs(self) -> Dict[str, TTSModelConfig]:
        return self._tts_configs
