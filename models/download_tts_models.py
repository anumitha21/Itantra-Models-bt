"""
Model Downloader for Offline Indic-TTS & MMS-TTS VITS Checkpoints.
Downloads VITS ONNX models and tokens for Hindi, Tamil, and Telugu from Hugging Face.
"""

import sys
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_backend.core.logging import get_logger

logger = get_logger("DownloadTTSModels")

MODELS_DIR = PROJECT_ROOT / "models" / "tts"

AI4BHARAT_REPO = "MatiasLin/sherpa-onnx-vits-rasa-13"

MMS_REPO = "willwade/mms-tts-multilingual-models-onnx"
MMS_LANG_MAP = {
    "hi": "hin",
    "ta": "tam",
    "te": "tel",
}


def download_ai4bharat_models(languages=None) -> None:
    target_langs = languages or ["hi", "ta", "te"]
    logger.info("=" * 60)
    logger.info("Downloading AI4Bharat Indic-TTS VITS Models...")
    logger.info("=" * 60)

    # Download AI4Bharat shared ONNX and tokens
    try:
        model_file = hf_hub_download(repo_id=AI4BHARAT_REPO, filename="model.onnx")
        tokens_file = hf_hub_download(repo_id=AI4BHARAT_REPO, filename="tokens.txt")

        model_bytes = Path(model_file).read_bytes()
        tokens_bytes = Path(tokens_file).read_bytes()

        for lang in target_langs:
            dest = MODELS_DIR / "ai4bharat" / lang
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "model.onnx").write_bytes(model_bytes)
            (dest / "tokens.txt").write_bytes(tokens_bytes)
            logger.info(f"    [OK] Prepared AI4Bharat Indic-TTS for [{lang.upper()}] at {dest}")
    except Exception as e:
        logger.error(f"    [FAIL] Failed to download AI4Bharat TTS: {e}")


def download_mms_models(languages=None) -> None:
    target_langs = languages or ["hi", "ta", "te"]
    logger.info("=" * 60)
    logger.info("Downloading Meta MMS-TTS VITS Models...")
    logger.info("=" * 60)

    for lang in target_langs:
        if lang not in MMS_LANG_MAP:
            continue
        mms_code = MMS_LANG_MAP[lang]
        dest = MODELS_DIR / "mms" / lang
        dest.mkdir(parents=True, exist_ok=True)
        logger.info(f"--> Downloading MMS-TTS [{lang.upper()} -> {mms_code}] from {MMS_REPO} to {dest}...")

        try:
            for filename in ["model.onnx", "tokens.txt"]:
                remote_path = f"{mms_code}/{filename}"
                f_path = hf_hub_download(repo_id=MMS_REPO, filename=remote_path)
                dest_file = dest / filename
                dest_file.write_bytes(Path(f_path).read_bytes())

            logger.info(f"    [OK] Downloaded MMS-TTS for [{lang.upper()}]")
        except Exception as e:
            logger.error(f"    [FAIL] Failed to download MMS TTS for [{lang.upper()}]: {e}")


def download_all_tts_models(languages=None) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    download_ai4bharat_models(languages=languages)
    download_mms_models(languages=languages)
    logger.info("=" * 60)
    logger.info("TTS Model Downloads Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Offline TTS Models (AI4Bharat & MMS).")
    parser.add_argument("--languages", nargs="+", choices=["hi", "ta", "te"], default=["hi", "ta", "te"], help="Languages to download.")
    parser.add_argument("--model", choices=["ai4bharat", "mms", "all"], default="all", help="Model family to download.")
    args = parser.parse_args()

    if args.model in ["ai4bharat", "all"]:
        download_ai4bharat_models(args.languages)
    if args.model in ["mms", "all"]:
        download_mms_models(args.languages)
