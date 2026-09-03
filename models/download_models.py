"""
Model Downloader for Offline IndicConformer STT Baseline.
Downloads INT8 quantized ONNX models for Hindi and English from HuggingFace.
"""

import sys
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

# Add parent directory to sys.path to allow importing src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import MODELS_DIR, HF_MODEL_REPO


def download_models(include_fp32: bool = False) -> None:
    """
    Download required INT8 ONNX models and tokens into the models directory.
    """
    print("=" * 60)
    print("AI4Bharat IndicConformer ONNX Model Downloader")
    print("=" * 60)
    print(f"Target Directory: {MODELS_DIR.resolve()}")
    print(f"HuggingFace Repository: {HF_MODEL_REPO}\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "hi").mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "en").mkdir(parents=True, exist_ok=True)

    # 1. Download tokens.txt
    tokens_target = MODELS_DIR / "tokens.txt"
    print("--> Downloading shared vocabulary tokens (tokens.txt)...")
    downloaded_tokens = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename="tokens.txt"
    )
    tokens_target.write_bytes(Path(downloaded_tokens).read_bytes())
    print(f"    Saved to: {tokens_target}\n")

    # 2. Download Hindi INT8 ONNX Model
    hi_int8_target = MODELS_DIR / "hi" / "model.int8.onnx"
    print("--> Downloading Hindi INT8 ONNX model (hi/model.int8.onnx)...")
    downloaded_hi = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename="hi/model.int8.onnx"
    )
    hi_int8_target.write_bytes(Path(downloaded_hi).read_bytes())
    print(f"    Saved to: {hi_int8_target}\n")

    # 3. Download English INT8 ONNX Model
    en_int8_target = MODELS_DIR / "en" / "model.int8.onnx"
    print("--> Downloading English INT8 ONNX model (en/model.int8.onnx)...")
    downloaded_en = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename="en/model.int8.onnx"
    )
    en_int8_target.write_bytes(Path(downloaded_en).read_bytes())
    print(f"    Saved to: {en_int8_target}\n")

    if include_fp32:
        # Download FP32 models from trysem/indicconformer-120m-onnx for comparison if requested
        fp32_repo = "trysem/indicconformer-120m-onnx"
        print(f"--> Downloading Hindi FP32 ONNX model from {fp32_repo}...")
        hi_fp32_target = MODELS_DIR / "hi" / "model.onnx"
        try:
            downloaded_hi_fp32 = hf_hub_download(
                repo_id=fp32_repo,
                filename="hi/model.onnx"
            )
            hi_fp32_target.write_bytes(Path(downloaded_hi_fp32).read_bytes())
            print(f"    Saved to: {hi_fp32_target}\n")
        except Exception as e:
            print(f"    Failed to download FP32 model: {e}")

    print("=" * 60)
    print("Download Complete! All required models are ready for offline inference.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download IndicConformer ONNX models.")
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Download optional FP32 ONNX models for configuration comparison."
    )
    args = parser.parse_args()
    download_models(include_fp32=args.fp32)
