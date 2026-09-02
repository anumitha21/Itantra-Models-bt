"""
Centralized Configuration for Offline IndicConformer STT Baseline.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories
MODELS_DIR = Path(os.getenv("STT_MODELS_DIR", PROJECT_ROOT / "models"))
TEST_AUDIO_DIR = Path(os.getenv("STT_TEST_AUDIO_DIR", PROJECT_ROOT / "test_audio"))
RESULTS_DIR = Path(os.getenv("STT_RESULTS_DIR", PROJECT_ROOT / "results"))
DEFAULT_RESULTS_CSV = Path(os.getenv("STT_RESULTS_CSV", RESULTS_DIR / "results.csv"))

# Audio settings
TARGET_SAMPLE_RATE = 16000  # IndicConformer requires 16kHz mono audio

# Supported initial languages
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "hindi": "hi",
    "hi": "hi",
    "english": "en",
    "en": "en",
}

# HuggingFace repository for pre-exported INT8 ONNX models
HF_MODEL_REPO = "parismitaglobalsolutions/indicconformer-sherpa-onnx"

# Default Model Paths (INT8 ONNX)
DEFAULT_MODEL_PATHS: Dict[str, Path] = {
    "hi": MODELS_DIR / "hi" / "model.int8.onnx",
    "en": MODELS_DIR / "en" / "model.int8.onnx",
}

# FP32 Alternative Paths (if model comparison is enabled)
FP32_MODEL_PATHS: Dict[str, Path] = {
    "hi": MODELS_DIR / "hi" / "model.onnx",
    "en": MODELS_DIR / "en" / "model.onnx",
}

# Shared Token Vocabulary Path
DEFAULT_TOKENS_PATH = MODELS_DIR / "tokens.txt"


@dataclass
class STTConfig:
    """Dataclass holding runtime configuration for STT transcription."""
    language: str = "hindi"
    model_path: Optional[Path] = None
    tokens_path: Optional[Path] = None
    num_threads: int = int(os.getenv("STT_NUM_THREADS", "2"))
    sample_rate: int = TARGET_SAMPLE_RATE
    results_csv: Path = DEFAULT_RESULTS_CSV
    precision: str = os.getenv("STT_PRECISION", "int8")  # "int8" or "fp32"

    def normalize_language(self) -> str:
        """Returns standard language code ('hi' or 'en')."""
        lang_lower = self.language.lower().strip()
        if lang_lower in SUPPORTED_LANGUAGES:
            return SUPPORTED_LANGUAGES[lang_lower]
        supported = ", ".join(list(SUPPORTED_LANGUAGES.keys()))
        raise ValueError(
            f"Unsupported language '{self.language}'. Supported languages: {supported}"
        )

    def get_model_file_path(self) -> Path:
        """Returns target model path based on language and precision."""
        if self.model_path:
            return Path(self.model_path)
        
        lang_code = self.normalize_language()
        if self.precision.lower() == "fp32":
            return FP32_MODEL_PATHS.get(lang_code, DEFAULT_MODEL_PATHS[lang_code])
        return DEFAULT_MODEL_PATHS[lang_code]

    def get_tokens_file_path(self) -> Path:
        """Returns vocabulary tokens file path."""
        if self.tokens_path:
            return Path(self.tokens_path)
        return DEFAULT_TOKENS_PATH
