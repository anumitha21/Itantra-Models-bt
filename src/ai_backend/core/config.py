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
class TTSModelConfig:
    name: str = "ai4bharat_vits"
    version: str = "1.0.0"
    language: str = "hi"
    path: str = "models/tts/ai4bharat/hi/model.onnx"
    tokens_path: str = "models/tts/ai4bharat/hi/tokens.txt"
    lexicon_path: Optional[str] = None
    data_dir: Optional[str] = None
    dict_dir: Optional[str] = None
    rule_fsts_path: Optional[str] = None
    rule_fars_path: Optional[str] = None
    quantization: str = "fp32"
    format: str = "onnx"
    architecture: str = "VITS"
    runtime: str = "sherpa-onnx"
    source: str = "AI4Bharat"
    expected_sample_rate: int = 22050
    speaker_id: int = 0
    noise_scale: float = 0.667
    noise_scale_w: float = 0.8
    length_scale: float = 1.0
    device: str = "cpu"

    def get_absolute_model_path(self, base_dir: Path = PROJECT_ROOT) -> Path:
        p = Path(self.path)
        return p if p.is_absolute() else base_dir / p

    def get_absolute_tokens_path(self, base_dir: Path = PROJECT_ROOT) -> Path:
        p = Path(self.tokens_path)
        return p if p.is_absolute() else base_dir / p

    def get_absolute_lexicon_path(self, base_dir: Path = PROJECT_ROOT) -> Optional[Path]:
        if not self.lexicon_path:
            return None
        p = Path(self.lexicon_path)
        return p if p.is_absolute() else base_dir / p

    def get_absolute_data_dir(self, base_dir: Path = PROJECT_ROOT) -> Optional[Path]:
        if not self.data_dir:
            return None
        p = Path(self.data_dir)
        return p if p.is_absolute() else base_dir / p

    def get_absolute_rule_fsts_path(self, base_dir: Path = PROJECT_ROOT) -> Optional[Path]:
        if not self.rule_fsts_path:
            return None
        p = Path(self.rule_fsts_path)
        return p if p.is_absolute() else base_dir / p

    def get_absolute_rule_fars_path(self, base_dir: Path = PROJECT_ROOT) -> Optional[Path]:
        if not self.rule_fars_path:
            return None
        p = Path(self.rule_fars_path)
        return p if p.is_absolute() else base_dir / p


@dataclass
class AppConfig:
    models_dir: Path = PROJECT_ROOT / "models"
    results_csv: Path = PROJECT_ROOT / "results" / "results.csv"
    tts_results_csv: Path = PROJECT_ROOT / "results" / "tts_results.csv"
    num_threads: int = 2
    sample_rate: int = 16000
    log_level: str = "INFO"
    stt_models: Dict[str, STTModelConfig] = field(default_factory=dict)
    tts_models: Dict[str, TTSModelConfig] = field(default_factory=dict)
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

        # 1. Parse STT models
        stt_models_cfg: Dict[str, STTModelConfig] = {}
        raw_stt = raw.get("models", {}).get("stt", {})

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

        # 2. Parse TTS models
        tts_models_cfg: Dict[str, TTSModelConfig] = {}
        raw_tts = raw.get("models", {}).get("tts", {})

        for model_key, item in raw_tts.items():
            lang = item.get("language", model_key.split("_")[-1] if "_" in model_key else model_key).lower()
            name = item.get("name", "ai4bharat_vits")
            tts_models_cfg[model_key.lower()] = TTSModelConfig(
                name=name,
                version=item.get("version", "1.0.0"),
                language=lang,
                path=item.get("path", f"models/tts/{name}/{lang}/model.onnx"),
                tokens_path=item.get("tokens_path", f"models/tts/{name}/{lang}/tokens.txt"),
                lexicon_path=item.get("lexicon_path"),
                data_dir=item.get("data_dir"),
                dict_dir=item.get("dict_dir"),
                rule_fsts_path=item.get("rule_fsts_path"),
                rule_fars_path=item.get("rule_fars_path"),
                quantization=item.get("quantization", "fp32"),
                format=item.get("format", "onnx"),
                architecture=item.get("architecture", "VITS"),
                runtime=item.get("runtime", "sherpa-onnx"),
                source=item.get("source", "AI4Bharat" if "ai4bharat" in name else "Meta AI"),
                expected_sample_rate=int(item.get("expected_sample_rate", 22050)),
                speaker_id=int(item.get("speaker_id", 0)),
                noise_scale=float(item.get("noise_scale", 0.667)),
                noise_scale_w=float(item.get("noise_scale_w", 0.8)),
                length_scale=float(item.get("length_scale", 1.0)),
                device=item.get("device", "cpu"),
            )

        return cls(
            num_threads=num_threads,
            sample_rate=sample_rate,
            log_level=log_level,
            stt_models=stt_models_cfg,
            tts_models=tts_models_cfg,
            raw_config=raw,
        )
