"""
Audio Utilities for Offline IndicConformer STT Baseline.
Handles WAV audio loading, format validation, mono conversion, duration calculation, and 16kHz resampling.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import soundfile as sf
import scipy.signal


class AudioProcessingError(Exception):
    """Custom exception raised for errors during audio loading or processing."""
    pass


def load_audio(
    audio_path: str | Path,
    target_sample_rate: int = 16000
) -> Tuple[np.ndarray, float, int]:
    """
    Load a WAV audio file, normalize channels, resample to target rate, and return samples & duration.

    Args:
        audio_path: Path to the WAV audio file.
        target_sample_rate: Target sampling rate required by IndicConformer (default: 16000).

    Returns:
        Tuple of (samples as float32 np.ndarray, duration in seconds, sample_rate).

    Raises:
        FileNotFoundError: If the audio file does not exist.
        AudioProcessingError: If the file is not a valid WAV or fails to load.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path.resolve()}")

    if not path.is_file():
        raise AudioProcessingError(f"Specified path is not a file: {path.resolve()}")

    try:
        data, sr = sf.read(str(path), dtype="float32")
    except Exception as e:
        raise AudioProcessingError(
            f"Failed to read WAV file '{path.name}'. Ensure it is a valid WAV audio file. Error: {e}"
        )

    if data.size == 0:
        raise AudioProcessingError(f"Audio file '{path.name}' is empty (0 samples).")

    # Multi-channel to mono conversion
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Calculate audio duration in seconds from original audio rate and length
    duration_sec = len(data) / float(sr)

    # Resample to target sample rate if necessary
    if sr != target_sample_rate:
        # Use polyphase filtering for fast high-quality resampling
        gcd = np.gcd(int(sr), int(target_sample_rate))
        up = int(target_sample_rate // gcd)
        down = int(sr // gcd)
        data = scipy.signal.resample_poly(data, up, down).astype(np.float32)
        sr = target_sample_rate

    # Ensure C-contiguous float32 array
    data = np.ascontiguousarray(data, dtype=np.float32)

    return data, duration_sec, sr
