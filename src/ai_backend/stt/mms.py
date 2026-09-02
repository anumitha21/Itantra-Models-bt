"""
Meta MMS ASR STT Engine Implementation (transformers / Wav2Vec2-CTC).
"""

import time
from typing import Optional, Dict
from pathlib import Path
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

from ai_backend.core.types import AudioInput, TranscriptionResult
from ai_backend.core.config import STTModelConfig, PROJECT_ROOT
from ai_backend.core.exceptions import ModelLoadError, InferenceError
from ai_backend.core.logging import get_logger
from ai_backend.models.model_metadata import ModelMetadata
from ai_backend.stt.base import BaseSTTEngine

logger = get_logger("MMSSTTEngine")

# ISO 639-1 to MMS ISO 639-3 language mapping
MMS_LANG_MAP: Dict[str, str] = {
    "hi": "hin",
    "hindi": "hin",
    "hin": "hin",
    "ta": "tam",
    "tamil": "tam",
    "tam": "tam",
    "te": "tel",
    "telugu": "tel",
    "tel": "tel",
    "en": "eng",
    "english": "eng",
    "eng": "eng",
}


class MMSSTTEngine(BaseSTTEngine):
    """
    Offline STT Engine for Meta MMS ASR models using HuggingFace Transformers.
    Runs on CPU with strictly pinned threads and adapters per target language.
    """

    def __init__(self, config: STTModelConfig, num_threads: int = 2):
        self.config = config
        self.num_threads = num_threads
        self.mms_lang = getattr(self.config, "mms_lang_code", None) or MMS_LANG_MAP.get(
            self.config.language.lower(), "hin"
        )
        self.model_id = self.config.version if "mms" in self.config.version else "facebook/mms-1b-all"
        self._processor: Optional[AutoProcessor] = None
        self._model: Optional[Wav2Vec2ForCTC] = None

    def _get_cache_dir(self) -> Path:
        p = Path(self.config.path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    def load(self) -> None:
        if self.is_loaded():
            return

        torch.set_num_threads(self.num_threads)
        cache_dir = self._get_cache_dir()
        logger.info(
            f"Loading MMS STT model [{self.model_id}] for language [{self.config.language.upper()} "
            f"-> {self.mms_lang}] (threads={self.num_threads}, cache={cache_dir})..."
        )

        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=str(cache_dir),
            )
            # Set target language vocabulary/adapter for tokenizer
            if hasattr(self._processor.tokenizer, "set_target_lang"):
                self._processor.tokenizer.set_target_lang(self.mms_lang)

            self._model = Wav2Vec2ForCTC.from_pretrained(
                self.model_id,
                cache_dir=str(cache_dir),
            )
            # Explicitly load specific language adapter for CTC head
            if hasattr(self._model, "load_adapter"):
                self._model.load_adapter(self.mms_lang)

            self._model.eval()
            self._model.to("cpu")
            logger.info(f"Successfully loaded MMS STT model with adapter [{self.mms_lang}] for [{self.config.language.upper()}].")
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load MMS ASR model '{self.model_id}' for language '{self.config.language}': {e}"
            )

    def unload(self) -> None:
        if self.is_loaded():
            logger.info(f"Unloading MMS STT model [{self.model_id}].")
            self._processor = None
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self._processor is not None and self._model is not None

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
            extra={"mms_lang_code": self.mms_lang, "runtime": "transformers-torch"},
        )

    def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        if not self.is_loaded():
            self.load()

        if audio is None or audio.samples.size == 0:
            raise ValueError("Input AudioInput samples are empty.")

        torch.set_num_threads(self.num_threads)
        # Ensure adapter is active
        if hasattr(self._processor.tokenizer, "set_target_lang"):
            self._processor.tokenizer.set_target_lang(self.mms_lang)
        if hasattr(self._model, "load_adapter"):
            self._model.load_adapter(self.mms_lang)

        start_time = time.perf_counter()

        try:
            # Process float32 waveform
            inputs = self._processor(
                audio.samples,
                sampling_rate=audio.sample_rate,
                return_tensors="pt"
            )
            with torch.no_grad():
                logits = self._model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self._processor.batch_decode(predicted_ids)[0]
            text = transcription.strip()
        except Exception as e:
            raise InferenceError(f"MMS ASR inference failed during transcription: {e}")

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
