"""
Unit Tests for Text-to-Speech (TTS) Models and Engines.
"""

import pytest
import numpy as np
from ai_backend.core.config import AppConfig, TTSModelConfig
from ai_backend.core.types import AudioInput
from ai_backend.core.exceptions import UnsupportedLanguage, UnsupportedModel
from ai_backend.tts.service import TTSServiceFactory
from ai_backend.tts.vits_engine import VitsTTSEngine
from ai_backend.models.model_manager import ModelManager


def test_tts_model_config_loading():
    config = AppConfig.load()
    assert "ai4bharat_vits_hi" in config.tts_models
    assert "mms_vits_hi" in config.tts_models

    cfg = config.tts_models["ai4bharat_vits_hi"]
    assert cfg.language == "hi"
    assert cfg.name == "ai4bharat_vits"
    assert cfg.expected_sample_rate in [22050, 24000, 16000]


def test_tts_service_factory_creation():
    config = AppConfig.load()
    cfg = config.tts_models["ai4bharat_vits_hi"]
    engine = TTSServiceFactory.create_engine(cfg, num_threads=2)
    assert isinstance(engine, VitsTTSEngine)
    assert not engine.is_loaded()


def test_vits_tts_engine_synthesize_and_resample():
    config = AppConfig.load()
    cfg = config.tts_models["ai4bharat_vits_hi"]
    engine = TTSServiceFactory.create_engine(cfg, num_threads=2)

    # Synthesize
    audio = engine.synthesize("नमस्ते", language="hi")
    assert isinstance(audio, AudioInput)
    assert audio.duration_sec > 0
    assert len(audio.samples) > 0
    assert audio.sample_rate in [22050, 24000, 16000]

    # Explicit Resampling to 16kHz for STT judge
    resampled_audio = audio.resample(target_sample_rate=16000)
    assert resampled_audio.sample_rate == 16000
    assert abs(resampled_audio.duration_sec - audio.duration_sec) < 0.05

    # Unload
    engine.unload()
    assert not engine.is_loaded()


def test_model_manager_tts_lifecycle():
    manager = ModelManager()

    # Load TTS
    tts_hi = manager.load_tts("hi", "ai4bharat_vits")
    assert tts_hi.is_loaded()
    assert manager._active_tts_key == "ai4bharat_vits_hi"

    # Loading a different TTS model should unload the previous one in non-benchmark mode
    tts_mms = manager.load_tts("hi", "mms_vits")
    assert tts_mms.is_loaded()
    assert not tts_hi.is_loaded()
    assert manager._active_tts_key == "mms_vits_hi"

    manager.unload_all()
    assert not tts_mms.is_loaded()
