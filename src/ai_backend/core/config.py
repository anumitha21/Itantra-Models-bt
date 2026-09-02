"""
Configuration Management Module for AI Backend Foundation.
Loads YAML configuration files with fallback to system defaults.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass
class STTModelConfig:
    name: str = "indicconformer"
    version: str = "1.0.0"
    language: str = "hi"
    path: str = "models/stt/hi/model.int8.onnx"
    tokens_path: str = "models/stt/tokens.txt"
    quantization: str = "int8"
    format: str = "onnx"
    architecture: str = "Conformer-CTC"
    runtime: str = "sherpa-onnx"
    source: str = "AI4Bharat"
    expected_sample_rate: int = 16000
    model_size: Optional[str] = None
    mms_lang_code: Optional[str] = None
    device: str = "cpu"

    def get_absolute_model_path(self, base_dir: Path = PROJECT_ROOT) -> Path:
        p = Path(self.path)
        return p if p.is_absolute() else base_dir / p

    def get_absolute_tokens_path(self, base_dir: Path = PROJECT_ROOT) -> Path:
        p = Path(self.tokens_path)
        return p if p.is_absolute() else base_dir / p


@dataclass
class AppConfig:
    models_dir: Path = PROJECT_ROOT / "models"
    results_csv: Path = PROJECT_ROOT / "results" / "results.csv"
    num_threads: int = 2
    sample_rate: int = 16000
    log_level: str = "INFO"
    stt_models: Dict[str, STTModelConfig] = field(default_factory=dict)
    raw_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> "AppConfig":
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        raw = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

        num_threads = int(os.getenv("STT_NUM_THREADS", raw.get("num_threads", 2)))
        sample_rate = int(raw.get("sample_rate", 16000))
        log_level = os.getenv("LOG_LEVEL", raw.get("log_level", "INFO"))

        stt_models_cfg: Dict[str, STTModelConfig] = {}
        raw_stt = raw.get("models", {}).get("stt", {})

        # Default fallback models if config file doesn't define them
        if not raw_stt:
            raw_stt = {
                "hi": {
                    "name": "indicconformer",
                    "language": "hi",
                    "path": "models/stt/hi/model.int8.onnx",
                    "tokens_path": "models/stt/tokens.txt",
                    "quantization": "int8",
                },
                "en": {
                    "name": "indicconformer",
                    "language": "en",
                    "path": "models/stt/en/model.int8.onnx",
                    "tokens_path": "models/stt/tokens.txt",
                    "quantization": "int8",
                },
            }

        for model_key, item in raw_stt.items():
            lang = item.get("language", model_key.split("_")[-1] if "_" in model_key else model_key).lower()
            name = item.get("name", "indicconformer")
            runtime = item.get("runtime", "sherpa-onnx" if name == "indicconformer" else "ctranslate2" if name == "whisper" else "transformers-torch")
            stt_models_cfg[model_key.lower()] = STTModelConfig(
                name=name,
                version=item.get("version", "1.0.0"),
                language=lang,
                path=item.get("path", f"models/stt/{lang}/model.int8.onnx"),
                tokens_path=item.get("tokens_path", "models/stt/tokens.txt"),
                quantization=item.get("quantization", "int8"),
                format=item.get("format", "onnx"),
                architecture=item.get("architecture", "Conformer-CTC"),
                runtime=runtime,
                source=item.get("source", "AI4Bharat"),
                expected_sample_rate=int(item.get("expected_sample_rate", 16000)),
                model_size=item.get("model_size"),
                mms_lang_code=item.get("mms_lang_code"),
                device=item.get("device", "cpu"),
            )

        return cls(
            num_threads=num_threads,
            sample_rate=sample_rate,
            log_level=log_level,
            stt_models=stt_models_cfg,
            raw_config=raw,
        )
