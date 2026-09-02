"""
Unit Tests for WhisperSTTEngine (src/ai_backend/stt/whisper_engine.py).
"""

import pytest
import numpy as np
from ai_backend.core.config import STTModelConfig
from ai_backend.core.types import AudioInput
from ai_backend.stt.whisper_engine import WhisperSTTEngine


def test_whisper_engine_init_and_metadata():
    cfg = STTModelConfig(
        name="whisper",
        model_size="tiny",
        language="hi",
        path="models/whisper/tiny",
        quantization="int8",
        format="ctranslate2",
        architecture="Encoder-Decoder-Transformer",
        runtime="ctranslate2",
    )
    engine = WhisperSTTEngine(config=cfg, num_threads=2)
    assert not engine.is_loaded()
    meta = engine.metadata()
    assert meta.name == "whisper"
    assert meta.language == "hi"
    assert meta.quantization == "int8"
    assert meta.extra.get("model_size") == "tiny"
    assert meta.extra.get("runtime") == "ctranslate2"


def test_whisper_engine_empty_audio_raises():
    cfg = STTModelConfig(
        name="whisper",
        model_size="tiny",
        language="hi",
        path="models/whisper/tiny",
        quantization="int8",
        format="ctranslate2",
    )
    engine = WhisperSTTEngine(config=cfg, num_threads=2)
    # Empty audio should raise ValueError
    empty_audio = AudioInput(samples=np.array([], dtype=np.float32), sample_rate=16000, duration_sec=0.0)
    with pytest.raises(ValueError):
        engine.transcribe(empty_audio)
