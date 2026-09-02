"""
Offline IndicConformer Transcriber Interface using sherpa-onnx.
"""

from pathlib import Path
import time
from typing import Dict, Any, Optional
import numpy as np
import sherpa_onnx

from src.config import STTConfig, TARGET_SAMPLE_RATE


class TranscriberError(Exception):
    """Base exception for transcriber errors."""
    pass


class ModelNotFoundError(TranscriberError, FileNotFoundError):
    """Raised when the specified ONNX model file does not exist."""
    pass


class TokensNotFoundError(TranscriberError, FileNotFoundError):
    """Raised when the specified vocabulary tokens file does not exist."""
    pass


class IndicConformerTranscriber:
    """
    Offline Speech-to-Text Transcriber using AI4Bharat IndicConformer ONNX model via sherpa-onnx.
    """

    def __init__(self, config: Optional[STTConfig] = None, language: str = "hindi"):
        """
        Initialize the offline recognizer with specified model and tokens.

        Args:
            config: Optional STTConfig object. If None, default configuration is used.
            language: Language name ('hindi' or 'english'). Overrides config.language if provided.
        """
        self.config = config or STTConfig(language=language)
        if language:
            self.config.language = language

        self.model_path = self.config.get_model_file_path()
        self.tokens_path = self.config.get_tokens_file_path()

        self._validate_paths()
        self.recognizer = self._init_recognizer()

    def _validate_paths(self) -> None:
        """Verify that model and vocabulary files exist on local disk."""
        if not self.model_path.exists():
            raise ModelNotFoundError(
                f"Model file not found at: {self.model_path.resolve()}\n"
                f"Please run 'python models/download_models.py' or check your models directory."
            )
        if not self.tokens_path.exists():
            raise TokensNotFoundError(
                f"Tokens file not found at: {self.tokens_path.resolve()}\n"
                f"Please run 'python models/download_models.py' or check your models directory."
            )

    def _init_recognizer(self) -> sherpa_onnx.OfflineRecognizer:
        """Instantiate sherpa-onnx OfflineRecognizer using NeMo CTC configuration."""
        try:
            recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=str(self.model_path),
                tokens=str(self.tokens_path),
                num_threads=self.config.num_threads,
                debug=False
            )
            return recognizer
        except Exception as e:
            raise TranscriberError(
                f"Failed to initialize sherpa-onnx recognizer with model '{self.model_path.name}': {e}"
            )

    def transcribe(
        self,
        audio_samples: np.ndarray,
        sample_rate: int = TARGET_SAMPLE_RATE
    ) -> Dict[str, Any]:
        """
        Transcribe audio samples locally and return the text transcription and inference timing.

        Args:
            audio_samples: 1D float32 numpy array containing audio samples.
            sample_rate: Sampling rate of the audio (default 16000).

        Returns:
            Dict containing:
                - 'transcription': Transcribed text string
                - 'inference_time_sec': Time taken for inference in seconds
        """
        if audio_samples is None or audio_samples.size == 0:
            raise ValueError("Audio samples array is empty or None.")

        start_time = time.perf_counter()

        try:
            stream = self.recognizer.create_stream()
            stream.accept_waveform(sample_rate, audio_samples)
            self.recognizer.decode_stream(stream)
            text = stream.result.text.strip()
        except Exception as e:
            raise TranscriberError(f"Inference failed during decoding: {e}")

        inference_time_sec = time.perf_counter() - start_time

        return {
            "transcription": text,
            "inference_time_sec": inference_time_sec,
        }
