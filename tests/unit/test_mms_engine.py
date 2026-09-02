"""
Unit Tests for MMSSTTEngine (src/ai_backend/stt/mms.py).
"""

import pytest
import numpy as np
from ai_backend.core.config import STTModelConfig
from ai_backend.core.types import AudioInput
from ai_backend.stt.mms import MMSSTTEngine, MMS_LANG_MAP


def test_mms_language_mapping():
    assert MMS_LANG_MAP["hi"] == "hin"
    assert MMS_LANG_MAP["hindi"] == "hin"
    assert MMS_LANG_MAP["ta"] == "tam"
    assert MMS_LANG_MAP["tamil"] == "tam"
    assert MMS_LANG_MAP["te"] == "tel"
    assert MMS_LANG_MAP["telugu"] == "tel"


def test_mms_engine_init_and_metadata():
    cfg = STTModelConfig(
        name="mms",
        version="facebook/mms-1b-all",
        language="ta",
        mms_lang_code="tam",
        path="models/mms/mms-1b-all",
        quantization="fp32",
        format="pytorch",
        architecture="Wav2Vec2-CTC",
        runtime="transformers-torch",
    )
    engine = MMSSTTEngine(config=cfg, num_threads=2)
    assert not engine.is_loaded()
    assert engine.mms_lang == "tam"
    meta = engine.metadata()
    assert meta.name == "mms"
    assert meta.language == "ta"
    assert meta.quantization == "fp32"
    assert meta.extra.get("mms_lang_code") == "tam"
    assert meta.extra.get("runtime") == "transformers-torch"


def test_mms_engine_empty_audio_raises():
    cfg = STTModelConfig(
        name="mms",
        version="facebook/mms-1b-all",
        language="hi",
        mms_lang_code="hin",
        path="models/mms/mms-1b-all",
        quantization="fp32",
    )
    engine = MMSSTTEngine(config=cfg, num_threads=2)
    empty_audio = AudioInput(samples=np.array([], dtype=np.float32), sample_rate=16000, duration_sec=0.0)
    with pytest.raises(ValueError):
        engine.transcribe(empty_audio)
