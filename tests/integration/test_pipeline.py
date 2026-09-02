"""
Integration Tests for SpeechPipeline and ModelManager.
"""

import pytest
from pathlib import Path
from ai_backend.core.types import AudioInput
from ai_backend.core.exceptions import UnsupportedLanguage, ServiceNotImplementedError
from ai_backend.models.model_manager import ModelManager
from ai_backend.pipeline.speech_pipeline import SpeechPipeline


def test_model_manager_load_and_unload():
    manager = ModelManager()

    engine_hi = manager.load_stt("hindi")
    assert engine_hi.is_loaded()
    assert manager._active_stt_lang == "hi"

    # Loading english should automatically unload hindi to meet RAM limits
    engine_en = manager.load_stt("english")
    assert engine_en.is_loaded()
    assert manager._active_stt_lang == "en"
    assert not engine_hi.is_loaded()

    manager.unload_all()
    assert not engine_en.is_loaded()


def test_pipeline_transcribe_hindi_wav():
    wav_path = Path("test_audio/hindi/test01.wav")
    assert wav_path.exists(), "test_audio/hindi/test01.wav must exist"

    audio = AudioInput.from_wav_file(wav_path)
    pipeline = SpeechPipeline()
    result = pipeline.transcribe(audio, language="hindi")

    assert result.success is True
    assert result.language == "hi"
    assert result.audio_duration_sec > 0
    assert result.inference_time_sec > 0
    assert result.rtf > 0
    assert "तीन" in result.text


def test_pipeline_unsupported_language_raises_error():
    pipeline = SpeechPipeline()
    audio = AudioInput.from_array([0.1] * 16000, sample_rate=16000)

    with pytest.raises(UnsupportedLanguage):
        pipeline.transcribe(audio, language="french")


def test_pipeline_tts_vad_stubs_raise_not_implemented():
    pipeline = SpeechPipeline()
    audio = AudioInput.from_array([0.1] * 16000, sample_rate=16000)

    with pytest.raises(ServiceNotImplementedError):
        pipeline.synthesize("hello", language="en")

    with pytest.raises(ServiceNotImplementedError):
        pipeline.vad_service.detect(audio)
