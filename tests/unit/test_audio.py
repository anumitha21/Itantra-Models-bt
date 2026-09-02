"""
Unit Tests for AudioInput DTO (src/ai_backend/core/types.py).
"""

import pytest
import numpy as np
import soundfile as sf
from ai_backend.core.types import AudioInput
from ai_backend.core.exceptions import InvalidAudio


def test_audio_input_from_array():
    samples = np.ones(16000, dtype=np.float32) * 0.5
    audio = AudioInput.from_array(samples, sample_rate=16000)

    assert audio.sample_rate == 16000
    assert pytest.approx(audio.duration_sec, rel=1e-3) == 1.0
    assert audio.samples.ndim == 1


def test_audio_input_from_wav_file(tmp_path):
    wav_file = tmp_path / "sample.wav"
    data = (0.5 * np.sin(np.linspace(0, 1.0, 16000))).astype(np.float32)
    sf.write(str(wav_file), data, 16000)

    audio = AudioInput.from_wav_file(wav_file, target_sample_rate=16000)
    assert audio.sample_rate == 16000
    assert pytest.approx(audio.duration_sec, rel=1e-2) == 1.0


def test_audio_input_invalid_file():
    with pytest.raises(FileNotFoundError):
        AudioInput.from_wav_file("non_existent.wav")


def test_audio_input_empty_samples():
    with pytest.raises(InvalidAudio):
        AudioInput.from_array(np.array([], dtype=np.float32))
