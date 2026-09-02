"""
AI4Bharat IndicConformer STT Engine Implementation (sherpa-onnx).
"""

import time
from typing import Optional
import sherpa_onnx

from ai_backend.core.types import AudioInput, TranscriptionResult
from ai_backend.core.config import STTModelConfig
from ai_backend.core.exceptions import ModelNotFound, TokensNotFound, ModelLoadError, InferenceError
from ai_backend.core.logging import get_logger
from ai_backend.models.model_metadata import ModelMetadata
from ai_backend.stt.base import BaseSTTEngine

logger = get_logger("IndicConformerSTTEngine")


class IndicConformerSTTEngine(BaseSTTEngine):
    """
    Offline STT Engine for AI4Bharat IndicConformer ONNX models using sherpa-onnx.
    """

    def __init__(self, config: STTModelConfig, num_threads: int = 2):
        self.config = config
        self.num_threads = num_threads
        self.model_path = self.config.get_absolute_model_path()
        self.tokens_path = self.config.get_absolute_tokens_path()
        self._recognizer: Optional[sherpa_onnx.OfflineRecognizer] = None

    def _validate_paths(self) -> None:
        if not self.model_path.exists():
            raise ModelNotFound(
                f"Model ONNX file not found at: {self.model_path.resolve()}\n"
                f"Run 'python scripts/download_models.py' to download model weights."
            )
        if not self.tokens_path.exists():
            raise TokensNotFound(
                f"Tokens vocabulary file not found at: {self.tokens_path.resolve()}\n"
                f"Run 'python scripts/download_models.py' to download tokens.txt."
            )

    def load(self) -> None:
        if self.is_loaded():
            return

        self._validate_paths()
        logger.info(f"Loading IndicConformer STT model [{self.config.language.upper()}] from {self.model_path.name}...")

        try:
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=str(self.model_path),
                tokens=str(self.tokens_path),
                num_threads=self.num_threads,
                debug=False
            )
            logger.info(f"Successfully loaded IndicConformer STT model [{self.config.language.upper()}].")
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load sherpa-onnx IndicConformer model for language '{self.config.language}': {e}"
            )

    def unload(self) -> None:
        if self.is_loaded():
            logger.info(f"Unloading IndicConformer STT model [{self.config.language.upper()}].")
            self._recognizer = None

    def is_loaded(self) -> bool:
        return self._recognizer is not None

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name=self.config.name,
            version=self.config.version,
            language=self.config.language,
            format=self.config.format,
            quantization=self.config.quantization,
            expected_sample_rate=self.config.expected_sample_rate,
            architecture=self.config.architecture,
            source=self.config.source,
        )

    def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        if not self.is_loaded():
            self.load()

        if audio is None or audio.samples.size == 0:
            raise ValueError("Input AudioInput samples are empty.")

        start_time = time.perf_counter()

        try:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(audio.sample_rate, audio.samples)
            self._recognizer.decode_stream(stream)
            text = stream.result.text.strip()
        except Exception as e:
            inference_time = time.perf_counter() - start_time
            raise InferenceError(f"Inference failed during decoding: {e}")

        inference_time_sec = time.perf_counter() - start_time
        rtf = inference_time_sec / audio.duration_sec if audio.duration_sec > 0 else 0.0

        return TranscriptionResult(
            text=text,
            language=self.config.language,
            audio_duration_sec=audio.duration_sec,
            inference_time_sec=inference_time_sec,
            rtf=rtf,
            model_name=self.config.name,
            model_version=self.config.version,
            quantization=self.config.quantization,
            success=True,
            error=None,
            metadata=self.metadata().to_dict(),
        )
