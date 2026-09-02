"""
OpenAI Whisper STT Engine Implementation using faster-whisper (CTranslate2).
"""

import time
from typing import Optional, Dict
from pathlib import Path
import faster_whisper

from ai_backend.core.types import AudioInput, TranscriptionResult
from ai_backend.core.config import STTModelConfig, PROJECT_ROOT
from ai_backend.core.exceptions import ModelLoadError, InferenceError
from ai_backend.core.logging import get_logger
from ai_backend.models.model_metadata import ModelMetadata
from ai_backend.stt.base import BaseSTTEngine

logger = get_logger("WhisperSTTEngine")

SCRIPT_PROMPTS: Dict[str, str] = {
    "hi": "नमस्ते, यह हिंदी में प्रतिलेखन है।",
    "hindi": "नमस्ते, यह हिंदी में प्रतिलेखन है।",
    "ta": "வணக்கம், இது தமிழ் உரை.",
    "tamil": "வணக்கம், இது தமிழ் உரை.",
    "te": "నమస్కారం, ఇది తెలుగు వ్రాత.",
    "telugu": "నమస్కారం, ఇది తెలుగు వ్రాత.",
}


class WhisperSTTEngine(BaseSTTEngine):
    """
    Offline STT Engine for OpenAI Whisper models using faster-whisper / CTranslate2.
    Configured with robust anti-hallucination parameters (VAD filtering, temperature=0,
    repetition penalty, no-repeat n-grams, task='transcribe', condition_on_previous_text=False).
    """

    def __init__(self, config: STTModelConfig, num_threads: int = 2):
        self.config = config
        self.num_threads = num_threads
        self.model_size = getattr(self.config, "model_size", None) or "tiny"
        self._model: Optional[faster_whisper.WhisperModel] = None

    def _get_download_root(self) -> Path:
        p = Path(self.config.path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    def load(self) -> None:
        if self.is_loaded():
            return

        compute_type = self.config.quantization if self.config.quantization in ["int8", "float32", "int8_float16"] else "int8"
        download_root = self._get_download_root()
        logger.info(
            f"Loading Whisper STT model [{self.model_size}] for [{self.config.language.upper()}] "
            f"(compute_type={compute_type}, threads={self.num_threads}, root={download_root})..."
        )

        try:
            self._model = faster_whisper.WhisperModel(
                model_size_or_path=self.model_size,
                device="cpu",
                device_index=0,
                compute_type=compute_type,
                cpu_threads=self.num_threads,
                num_workers=1,
                download_root=str(download_root),
            )
            logger.info(f"Successfully loaded Whisper STT model [{self.model_size}].")
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load faster-whisper model '{self.model_size}' for language '{self.config.language}': {e}"
            )

    def unload(self) -> None:
        if self.is_loaded():
            logger.info(f"Unloading Whisper STT model [{self.model_size}].")
            self._model = None

    def is_loaded(self) -> bool:
        return self._model is not None

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
            extra={"model_size": self.model_size, "runtime": "ctranslate2"},
        )

    def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        if not self.is_loaded():
            self.load()

        if audio is None or audio.samples.size == 0:
            raise ValueError("Input AudioInput samples are empty.")

        start_time = time.perf_counter()

        try:
            prompt = SCRIPT_PROMPTS.get(self.config.language.lower(), None)
            segments, info = self._model.transcribe(
                audio.samples,
                language=self.config.language,
                task="transcribe",
                initial_prompt=prompt,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                hallucination_silence_threshold=2.0,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
            )
            text_segments = [seg.text for seg in segments]
            text = " ".join(text_segments).strip()
        except Exception as e:
            raise InferenceError(f"Whisper inference failed during transcription: {e}")

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
