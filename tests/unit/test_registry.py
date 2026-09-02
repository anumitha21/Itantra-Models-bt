"""
Unit Tests for Model Registry (src/ai_backend/models/model_registry.py).
"""

import pytest
from ai_backend.models.model_registry import ModelRegistry
from ai_backend.core.exceptions import UnsupportedLanguage, UnsupportedModel


def test_registry_language_normalization():
    registry = ModelRegistry()
    assert registry.normalize_language("hindi") == "hi"
    assert registry.normalize_language("ENGLISH") == "en"
    assert registry.normalize_language("hi") == "hi"
    assert registry.normalize_language("tamil") == "ta"
    assert registry.normalize_language("ta") == "ta"
    assert registry.normalize_language("telugu") == "te"
    assert registry.normalize_language("te") == "te"

    with pytest.raises(UnsupportedLanguage):
        registry.normalize_language("french")


def test_registry_metadata():
    registry = ModelRegistry()
    meta_hi = registry.get_stt_metadata("hindi")
    assert meta_hi.language == "hi"
    assert meta_hi.quantization == "int8"
    assert meta_hi.format == "onnx"


def test_registry_multi_model_lookup():
    registry = ModelRegistry()
    cfg_indic = registry.get_stt_config("hi", model_name="indicconformer")
    assert cfg_indic.name == "indicconformer"
    assert cfg_indic.language == "hi"

    cfg_whisper_tiny = registry.get_stt_config("ta", model_name="whisper_tiny")
    assert cfg_whisper_tiny.name == "whisper"
    assert cfg_whisper_tiny.model_size == "tiny"
    assert cfg_whisper_tiny.language == "ta"

    cfg_whisper_small = registry.get_stt_config("te", model_name="whisper_small")
    assert cfg_whisper_small.name == "whisper"
    assert cfg_whisper_small.model_size == "small"
    assert cfg_whisper_small.language == "te"

    cfg_mms = registry.get_stt_config("hi", model_name="mms")
    assert cfg_mms.name == "mms"
    assert cfg_mms.language == "hi"
