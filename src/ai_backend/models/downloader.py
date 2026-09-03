"""
Model Downloader and Verification Utility.
Ensures ONNX weights and token files are downloaded once and cached persistently on disk.
Subsequent calls return immediately with zero network overhead.
"""

from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download

from ai_backend.core.config import PROJECT_ROOT
from ai_backend.core.logging import get_logger

logger = get_logger("ModelDownloader")

# HuggingFace Repositories
INDICCONFORMER_HF_REPO = "parismitaglobalsolutions/indicconformer-sherpa-onnx"
AI4BHARAT_TTS_HF_REPO = "MatiasLin/sherpa-onnx-vits-rasa-13"
MMS_TTS_HF_REPO = "willwade/mms-tts-multilingual-models-onnx"

MMS_TTS_LANG_MAP = {
    "hi": "hin",
    "hindi": "hin",
    "ta": "tam",
    "tamil": "tam",
    "te": "tel",
    "telugu": "tel",
}


def ensure_indicconformer_model(language: str, models_root: Optional[Path] = None) -> Path:
    """
    Ensure IndicConformer INT8 ONNX model and shared tokens.txt exist on disk.
    Downloads them once from HuggingFace if absent.
    """
    base = models_root or (PROJECT_ROOT / "models" / "stt")
    lang_code = language.lower().strip()
    if lang_code in ["hindi", "hin"]:
        lang_code = "hi"
    elif lang_code in ["tamil", "tam"]:
        lang_code = "ta"
    elif lang_code in ["telugu", "tel"]:
        lang_code = "te"
    elif lang_code in ["english", "eng"]:
        lang_code = "en"

    base.mkdir(parents=True, exist_ok=True)
    tokens_path = base / "tokens.txt"
    model_dir = base / lang_code
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.int8.onnx"

    # 1. Check & download tokens.txt
    if not tokens_path.exists() or tokens_path.stat().st_size == 0:
        logger.info(f"Downloading shared tokens.txt from {INDICCONFORMER_HF_REPO} to {tokens_path}...")
        try:
            downloaded = hf_hub_download(
                repo_id=INDICCONFORMER_HF_REPO,
                filename="tokens.txt",
            )
            tokens_path.write_bytes(Path(downloaded).read_bytes())
            logger.info(f"Saved shared tokens to {tokens_path}.")
        except Exception as e:
            logger.error(f"Failed to download tokens.txt: {e}")
            raise

    # 2. Check & download model.int8.onnx
    if not model_path.exists() or model_path.stat().st_size == 0:
        logger.info(f"Downloading IndicConformer [{lang_code.upper()}] model from {INDICCONFORMER_HF_REPO}...")
        try:
            downloaded = hf_hub_download(
                repo_id=INDICCONFORMER_HF_REPO,
                filename=f"{lang_code}/model.int8.onnx",
            )
            model_path.write_bytes(Path(downloaded).read_bytes())
            logger.info(f"Saved IndicConformer [{lang_code.upper()}] model to {model_path}.")
        except Exception as e:
            logger.error(f"Failed to download IndicConformer [{lang_code}] model: {e}")
            raise

    return model_path


def ensure_tts_model(language: str, model_name: str, models_root: Optional[Path] = None) -> Path:
    """
    Ensure VITS TTS model (AI4Bharat or Meta MMS) and tokens exist on disk.
    Downloads them once from HuggingFace if absent.
    """
    base = models_root or (PROJECT_ROOT / "models" / "tts")
    lang_code = language.lower().strip()
    if lang_code in ["hindi", "hin"]:
        lang_code = "hi"
    elif lang_code in ["tamil", "tam"]:
        lang_code = "ta"
    elif lang_code in ["telugu", "tel"]:
        lang_code = "te"

    m_name = model_name.lower().strip()

    if "ai4bharat" in m_name:
        target_dir = base / "ai4bharat" / lang_code
        target_dir.mkdir(parents=True, exist_ok=True)
        model_path = target_dir / "model.onnx"
        tokens_path = target_dir / "tokens.txt"

        if not model_path.exists() or not tokens_path.exists() or model_path.stat().st_size == 0:
            logger.info(f"Downloading AI4Bharat Indic-TTS VITS model from {AI4BHARAT_TTS_HF_REPO}...")
            try:
                dl_model = hf_hub_download(repo_id=AI4BHARAT_TTS_HF_REPO, filename="model.onnx")
                dl_tokens = hf_hub_download(repo_id=AI4BHARAT_TTS_HF_REPO, filename="tokens.txt")
                model_path.write_bytes(Path(dl_model).read_bytes())
                tokens_path.write_bytes(Path(dl_tokens).read_bytes())
                logger.info(f"Saved AI4Bharat Indic-TTS for [{lang_code.upper()}] to {target_dir}.")
            except Exception as e:
                logger.error(f"Failed to download AI4Bharat Indic-TTS model: {e}")
                raise
        return model_path

    elif "mms" in m_name:
        target_dir = base / "mms" / lang_code
        target_dir.mkdir(parents=True, exist_ok=True)
        model_path = target_dir / "model.onnx"
        tokens_path = target_dir / "tokens.txt"

        mms_lang = MMS_TTS_LANG_MAP.get(lang_code, "hin")
        if not model_path.exists() or not tokens_path.exists() or model_path.stat().st_size == 0:
            logger.info(f"Downloading Meta MMS-TTS VITS [{mms_lang}] from {MMS_TTS_HF_REPO}...")
            try:
                for fname in ["model.onnx", "tokens.txt"]:
                    remote_file = f"{mms_lang}/{fname}"
                    dl_file = hf_hub_download(repo_id=MMS_TTS_HF_REPO, filename=remote_file)
                    (target_dir / fname).write_bytes(Path(dl_file).read_bytes())
                logger.info(f"Saved Meta MMS-TTS for [{lang_code.upper()} -> {mms_lang}] to {target_dir}.")
            except Exception as e:
                logger.error(f"Failed to download Meta MMS-TTS model for {lang_code}: {e}")
                raise
        return model_path

    return base / m_name / lang_code / "model.onnx"
