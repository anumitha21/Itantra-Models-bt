"""
VITS Offline TTS Engine Module based on sherpa-onnx.
Supports AI4Bharat Indic-TTS and Meta MMS-TTS ONNX checkpoints.
"""

import os
import gc
from pathlib import Path
from typing import Optional
import numpy as np
import sherpa_onnx

from ai_backend.core.config import TTSModelConfig
from ai_backend.core.exceptions import ModelLoadError, InferenceError
from ai_backend.core.types import AudioInput
from ai_backend.core.logging import get_logger
from ai_backend.tts.base import BaseTTSEngine

logger = get_logger("VitsTTSEngine")


class VitsTTSEngine(BaseTTSEngine):
    """
    Offline VITS Text-to-Speech Engine utilizing sherpa-onnx.
    Parameterized by TTSModelConfig to drive both AI4Bharat Indic-TTS and MMS-TTS models.
    """

    def __init__(self, config: TTSModelConfig, num_threads: int = 2):
        self.config = config
        self.num_threads = num_threads
        self._engine: Optional[sherpa_onnx.OfflineTts] = None

    def load(self) -> None:
        """
        Load VITS ONNX model and tokenizer/lexicon/rules into memory.
        """
        if self._engine is not None:
            return

        model_path = self.config.get_absolute_model_path()
        tokens_path = self.config.get_absolute_tokens_path()

        if not model_path.exists() or not tokens_path.exists():
            try:
                from ai_backend.models.downloader import ensure_tts_model
                ensure_tts_model(self.config.language, self.config.name)
            except Exception as e:
                logger.warning(f"Auto-download attempt for VITS TTS model failed: {e}")

        if not model_path.exists():
            raise ModelLoadError(
                f"VITS TTS model file not found at: {model_path.resolve()}"
            )
        if not tokens_path.exists():
            raise ModelLoadError(
                f"VITS TTS tokens file not found at: {tokens_path.resolve()}"
            )

        lexicon_path = self.config.get_absolute_lexicon_path()
        data_dir = self.config.get_absolute_data_dir()
        rule_fsts_path = self.config.get_absolute_rule_fsts_path()
        rule_fars_path = self.config.get_absolute_rule_fars_path()

        logger.info(
            f"Loading VITS TTS engine [{self.config.name}] for [{self.config.language.upper()}] "
            f"(threads={self.num_threads}, model={model_path.name})..."
        )

        try:
            vits_config = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(model_path.resolve()),
                tokens=str(tokens_path.resolve()),
                lexicon=str(lexicon_path.resolve()) if (lexicon_path and lexicon_path.exists()) else "",
                data_dir=str(data_dir.resolve()) if (data_dir and data_dir.exists()) else "",
                dict_dir="",
                noise_scale=self.config.noise_scale,
                noise_scale_w=self.config.noise_scale_w,
                length_scale=self.config.length_scale,
            )

            model_config = sherpa_onnx.OfflineTtsModelConfig(
                vits=vits_config,
                num_threads=self.num_threads,
                debug=False,
                provider="cpu",
            )

            tts_config = sherpa_onnx.OfflineTtsConfig(
                model=model_config,
                rule_fsts=str(rule_fsts_path.resolve()) if (rule_fsts_path and rule_fsts_path.exists()) else "",
                rule_fars=str(rule_fars_path.resolve()) if (rule_fars_path and rule_fars_path.exists()) else "",
                max_num_sentences=1,
            )

            self._engine = sherpa_onnx.OfflineTts(tts_config)
            logger.info(
                f"Successfully loaded VITS TTS model [{self.config.name}] for [{self.config.language.upper()}] "
                f"(sample_rate={self._engine.sample_rate}Hz, num_speakers={self._engine.num_speakers})."
            )
        except Exception as e:
            self._engine = None
            raise ModelLoadError(
                f"Failed to initialize sherpa-onnx OfflineTts for {self.config.name} ({self.config.language}): {e}"
            ) from e

    def unload(self) -> None:
        """
        Unload VITS TTS model and release RAM.
        """
        if self._engine is not None:
            logger.info(
                f"Unloading VITS TTS model [{self.config.name}] for [{self.config.language.upper()}]."
            )
            self._engine = None
            gc.collect()

    def is_loaded(self) -> bool:
        """
        Check whether the VITS TTS engine is loaded in memory.
        """
        return self._engine is not None

    def synthesize(self, text: str, language: Optional[str] = None, sid: Optional[int] = None, speed: float = 1.0) -> AudioInput:
        """
        Synthesize input text string into AudioInput.
        """
        if self._engine is None:
            self.load()

        if not text or not text.strip():
            # Return silence for empty string
            sr = self._engine.sample_rate if self._engine else self.config.expected_sample_rate
            return AudioInput.from_numpy(np.zeros(sr // 2, dtype=np.float32), sample_rate=sr)

        speaker_id = sid if sid is not None else self.config.speaker_id

        try:
            audio = self._engine.generate(
                text=text.strip(),
                sid=speaker_id,
                speed=speed,
            )
            samples = np.array(audio.samples, dtype=np.float32)
            sample_rate = int(audio.sample_rate)

            return AudioInput.from_numpy(
                samples=samples,
                sample_rate=sample_rate,
            )
        except Exception as e:
            raise InferenceError(
                f"VITS TTS synthesis failed for '{text[:30]}...' on model {self.config.name}: {e}"
            ) from e
