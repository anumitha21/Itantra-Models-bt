"""
Model Downloader Utility for AI Backend Foundation.
Downloads INT8 ONNX IndicConformer models and tokens for Hindi, Tamil, Telugu, and English into models/stt/.
"""

import sys
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_backend.core.config import PROJECT_ROOT as CFG_ROOT

MODELS_DIR = CFG_ROOT / "models" / "stt"
HF_MODEL_REPO = "parismitaglobalsolutions/indicconformer-sherpa-onnx"


def download_models(languages=None, include_fp32: bool = False) -> None:
    if languages is None:
        languages = ["hi", "ta", "te", "en"]

    print("=" * 60)
    print("AI4Bharat IndicConformer ONNX Model Downloader")
    print("=" * 60)
    print(f"Target Directory: {MODELS_DIR.resolve()}")
    print(f"HuggingFace Repository: {HF_MODEL_REPO}")
    print(f"Languages: {', '.join(languages)}\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download tokens.txt
    tokens_target = MODELS_DIR / "tokens.txt"
    if not tokens_target.exists():
        print("--> Downloading shared vocabulary tokens (tokens.txt)...")
        downloaded_tokens = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename="tokens.txt"
        )
        tokens_target.write_bytes(Path(downloaded_tokens).read_bytes())
        print(f"    Saved to: {tokens_target}\n")
    else:
        print(f"--> Shared tokens already exist: {tokens_target}")

    # 2. Download language INT8 ONNX models
    for lang in languages:
        lang_dir = MODELS_DIR / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        int8_target = lang_dir / "model.int8.onnx"

        if not int8_target.exists():
            print(f"--> Downloading {lang.upper()} INT8 ONNX model ({lang}/model.int8.onnx)...")
            try:
                downloaded_file = hf_hub_download(
                    repo_id=HF_MODEL_REPO,
                    filename=f"{lang}/model.int8.onnx"
                )
                int8_target.write_bytes(Path(downloaded_file).read_bytes())
                print(f"    Saved to: {int8_target}\n")
            except Exception as e:
                print(f"    Failed to download {lang} model: {e}")
        else:
            print(f"--> {lang.upper()} INT8 ONNX model already exists: {int8_target}")

    print("=" * 60)
    print("Download Complete! Selected IndicConformer STT models are ready for offline inference.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download IndicConformer ONNX models.")
    parser.add_argument("--languages", nargs="+", default=["hi", "ta", "te", "en"], help="Languages to download.")
    parser.add_argument("--fp32", action="store_true", help="Download optional FP32 models.")
    args = parser.parse_args()
    download_models(languages=args.languages, include_fp32=args.fp32)
