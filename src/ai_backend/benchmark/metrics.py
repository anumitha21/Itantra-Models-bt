"""
Decoupled Benchmark Metrics Module.
Calculates Word Error Rate (WER), Character Error Rate (CER),
Process RSS Memory, VRAM, and Model Disk Footprint.
"""

import os
import psutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import jiwer


import re
import unicodedata


def normalize_text(text: Optional[str]) -> str:
    """
    Standard text normalization for ASR evaluation (Indic & Latin):
    - Unicode NFC normalization
    - Lowercase conversion
    - Stripping punctuation and Indic dandas (।, ॥)
    - Collapsing consecutive whitespaces
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.lower()
    # Strip standard punctuation and symbols
    text = re.sub(r"[\.,\?!;:\"'()\[\]\{\}\-—_/\\]", " ", text)
    # Strip Indic punctuation (Devanagari danda / double danda)
    text = re.sub(r"[\u0964\u0965]", " ", text)
    # Collapse multiple whitespace characters
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_accuracy_metrics(
    reference: Optional[str],
    hypothesis: str
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute Word Error Rate (WER) and Character Error Rate (CER) after text normalization.
    Returns (None, None) if reference text is None or empty string.
    """
    if not reference or not str(reference).strip():
        return None, None

    ref_clean = normalize_text(reference)
    hyp_clean = normalize_text(hypothesis)

    if not ref_clean:
        return None, None

    if not hyp_clean:
        return 1.0, 1.0

    try:
        wer_val = float(jiwer.wer(ref_clean, hyp_clean))
        cer_val = float(jiwer.cer(ref_clean, hyp_clean))
        return wer_val, cer_val
    except Exception:
        return None, None


def get_process_rss_mb() -> float:
    """
    Get current process Resident Set Size (RSS) memory in megabytes.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)


def get_gpu_vram_mb() -> Optional[float]:
    """
    Get GPU VRAM allocated in MB if CUDA is available on the dev machine.
    Returns None if GPU is not available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    except Exception:
        pass
    return None


def get_model_size_mb(path: str | Path) -> float:
    """
    Compute total file or directory size on disk in megabytes.
    """
    p = Path(path)
    if not p.exists():
        return 0.0

    if p.is_file():
        return p.stat().st_size / (1024.0 * 1024.0)

    total_bytes = 0
    for entry in p.rglob("*"):
        if entry.is_file():
            total_bytes += entry.stat().st_size
    return total_bytes / (1024.0 * 1024.0)


def calculate_weighted_overall_score(
    wer: Optional[float],
    latency_warm_sec: float,
    ram_mb: float,
    model_size_mb: float,
    energy_joules: Optional[float] = None,
    # Normalization baselines
    max_latency_ref: float = 10.0,
    max_ram_ref: float = 4000.0,
    max_size_ref: float = 4000.0,
) -> Optional[float]:
    """
    Compute optional composite benchmark score (0 to 100, higher is better).
    Default Weights:
      - Accuracy (WER): 40%
      - Latency: 25%
      - RAM Memory: 15%
      - Model Size: 10%
      - Energy: 10% (renormalized to 0% if unmeasured)
    """
    if wer is None:
        return None

    # Accuracy subscore: 1.0 - wer (clamped between 0 and 1)
    acc_subscore = max(0.0, min(1.0, 1.0 - wer))

    # Efficiency subscores: lower is better (clamped 0 to 1)
    lat_subscore = max(0.0, min(1.0, 1.0 - (latency_warm_sec / max_latency_ref)))
    ram_subscore = max(0.0, min(1.0, 1.0 - (ram_mb / max_ram_ref)))
    size_subscore = max(0.0, min(1.0, 1.0 - (model_size_mb / max_size_ref)))

    if energy_joules is not None:
        energy_subscore = max(0.0, min(1.0, 1.0 - (energy_joules / 100.0)))
        score = (
            0.40 * acc_subscore
            + 0.25 * lat_subscore
            + 0.15 * ram_subscore
            + 0.10 * size_subscore
            + 0.10 * energy_subscore
        ) * 100.0
    else:
        # Renormalize weights: sum = 0.40 + 0.25 + 0.15 + 0.10 = 0.90
        w_acc = 0.40 / 0.90
        w_lat = 0.25 / 0.90
        w_ram = 0.15 / 0.90
        w_size = 0.10 / 0.90
        score = (
            w_acc * acc_subscore
            + w_lat * lat_subscore
            + w_ram * ram_subscore
            + w_size * size_subscore
        ) * 100.0

    return round(score, 2)
