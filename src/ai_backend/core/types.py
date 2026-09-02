"""
Core Data Types and Data Transfer Objects (DTOs) for AI Backend Foundation.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np
import soundfile as sf
import scipy.signal

from ai_backend.core.exceptions import InvalidAudio


@dataclass
class AudioInput:
    """
    Generic Audio Input DTO. Decouples STT engines from specific audio file formats or hardware streams.
    """
    samples: np.ndarray
    sample_rate: int
    duration_sec: float

    @classmethod
    def from_wav_file(cls, path: str | Path, target_sample_rate: int = 16000) -> "AudioInput":
        """
        Load audio from a WAV file, normalize channels, resample if necessary, and return AudioInput.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path.resolve()}")

        try:
            data, sr = sf.read(str(file_path), dtype="float32")
        except Exception as e:
            raise InvalidAudio(f"Failed to read WAV audio file '{file_path.name}': {e}")

        if data.size == 0:
            raise InvalidAudio(f"Audio file '{file_path.name}' is empty.")

        # Convert multi-channel to mono
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        duration_sec = float(len(data)) / float(sr)

        # Resample to target sample rate if needed
        if sr != target_sample_rate:
            gcd = np.gcd(int(sr), int(target_sample_rate))
            up = int(target_sample_rate // gcd)
            down = int(sr // gcd)
            data = scipy.signal.resample_poly(data, up, down).astype(np.float32)
            sr = target_sample_rate

        data = np.ascontiguousarray(data, dtype=np.float32)
        return cls(samples=data, sample_rate=sr, duration_sec=duration_sec)

    @classmethod
    def from_array(cls, samples: np.ndarray | list, sample_rate: int = 16000) -> "AudioInput":
        """
        Create AudioInput from raw numpy array or sample list.
        """
        if samples is None:
            raise InvalidAudio("Audio samples input is None.")

        data = np.asarray(samples, dtype=np.float32)
        if data.size == 0:
            raise InvalidAudio("Audio samples array is empty.")

        if data.ndim > 1:
            data = np.mean(data, axis=1)

        duration_sec = float(len(data)) / float(sample_rate)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return cls(samples=data, sample_rate=sample_rate, duration_sec=duration_sec)


@dataclass
class TranscriptionResult:
    """
    Structured Transcription Result DTO returned by STT engines.
    """
    text: str
    language: str
    audio_duration_sec: float
    inference_time_sec: float
    rtf: float
    model_name: str = "indicconformer"
    model_version: str = "1.0.0"
    quantization: str = "int8"
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize TranscriptionResult to dictionary representation.
        """
        return {
            "text": self.text,
            "language": self.language,
            "audio_duration_sec": self.audio_duration_sec,
            "inference_time_sec": self.inference_time_sec,
            "rtf": self.rtf,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "quantization": self.quantization,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }
