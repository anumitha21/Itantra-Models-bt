"""
Unit Tests for Configuration (src/ai_backend/core/config.py).
"""

import pytest
from pathlib import Path
from ai_backend.core.config import AppConfig, STTModelConfig


def test_app_config_defaults():
    cfg = AppConfig.load()
    assert cfg.num_threads == 2
    assert cfg.sample_rate == 16000
    assert "hi" in cfg.stt_models
    assert "en" in cfg.stt_models


def test_stt_model_config_paths():
    stt_cfg = STTModelConfig(
        language="hi",
        path="models/stt/hi/model.int8.onnx",
        tokens_path="models/stt/tokens.txt"
    )
    abs_model = stt_cfg.get_absolute_model_path()
    abs_tokens = stt_cfg.get_absolute_tokens_path()

    assert abs_model.name == "model.int8.onnx"
    assert abs_tokens.name == "tokens.txt"
