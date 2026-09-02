"""
Decoupled Benchmark Metrics Module.
Calculates Word Error Rate (WER), Character Error Rate (CER),
Process RSS Memory, VRAM, and Model Disk Footprint.
"""

import os
import time
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
    Compute composite benchmark score (0 to 100, higher is better).
    Resource-cost proxies: RAM (RSS), disk model size, and inference latency/RTF.
    Weights:
      - Accuracy (WER): 45%
      - Latency / RTF: 25%
      - Peak RAM: 18%
      - Model Size: 12%
    """
    if wer is None:
        return None

    # Accuracy subscore: 1.0 - wer (clamped between 0 and 1)
    acc_subscore = max(0.0, min(1.0, 1.0 - wer))

    # Efficiency subscores: lower is better (clamped 0 to 1)
    lat_subscore = max(0.0, min(1.0, 1.0 - (latency_warm_sec / max_latency_ref)))
    ram_subscore = max(0.0, min(1.0, 1.0 - (ram_mb / max_ram_ref)))
    size_subscore = max(0.0, min(1.0, 1.0 - (model_size_mb / max_size_ref)))

    score = (
        0.45 * acc_subscore
        + 0.25 * lat_subscore
        + 0.18 * ram_subscore
        + 0.12 * size_subscore
    ) * 100.0

    return round(score, 2)


compute_composite_score = calculate_weighted_overall_score


class EnergyBenchmarkTracker:
    """
    Hardware and runtime energy tracker for speech inference benchmarking.
    Measures total Joules consumed using codecarbon (hardware RAPL registers)
    with seamless fallback to CPU platform TDP profiling.
    """
    def __init__(self, measure_power_secs: int = 1):
        self.measure_power_secs = measure_power_secs
        self.tracker = None
        self.method = "none"
        self._start_time = 0.0
        self._energy_kwh = 0.0

    def start(self):
        self._start_time = time.perf_counter()
        try:
            import codecarbon
            self.tracker = codecarbon.EmissionsTracker(
                measure_power_secs=self.measure_power_secs,
                save_to_file=False,
                log_level="error",
            )
            self.tracker.start()
            self.method = "codecarbon_rapl" if getattr(self.tracker, "_is_rapl_available", False) else "codecarbon_tdp_model"
        except Exception:
            self.tracker = None
            self.method = "hardware_tdp_model"

    def stop(self) -> Tuple[float, str]:
        """
        Stop energy tracking and return (total_energy_joules, power_measurement_method).
        """
        elapsed_sec = max(0.001, time.perf_counter() - self._start_time)
        energy_joules = 0.0

        if self.tracker is not None:
            try:
                self.tracker.stop()
                if hasattr(self.tracker, "_total_energy") and hasattr(self.tracker._total_energy, "kWh"):
                    self._energy_kwh = float(self.tracker._total_energy.kWh)
                elif hasattr(self.tracker, "final_emissions_data") and hasattr(self.tracker.final_emissions_data, "energy_consumed"):
                    self._energy_kwh = float(self.tracker.final_emissions_data.energy_consumed)
                energy_joules = self._energy_kwh * 3.6e6  # 1 kWh = 3.6e6 Joules
            except Exception:
                pass

        if energy_joules <= 0.0:
            # Platform TDP model fallback: active CPU core package power (~28W typical laptop TDP)
            active_tdp_watts = 28.0
            energy_joules = active_tdp_watts * elapsed_sec
            if self.method == "none":
                self.method = "hardware_tdp_model"

        return round(energy_joules, 4), self.method
