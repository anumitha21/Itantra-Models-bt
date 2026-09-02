"""
Benchmark Dataset Runner for Reproducible Multi-Model STT Evaluation.
Iterates over models x languages x conditions defined in manifest.csv,
measures cold & warm latency, RTF, RAM RSS, disk model size, WER, and CER,
and appends results to results/results.csv with explicit runtime/precision provenance.
"""

import os
import sys
import csv
import time
import uuid
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_backend.core.config import AppConfig, PROJECT_ROOT as CFG_ROOT
from ai_backend.core.types import AudioInput
from ai_backend.core.logging import get_logger
from ai_backend.models.model_manager import ModelManager
from ai_backend.benchmark.metrics import (
    compute_accuracy_metrics,
    get_process_rss_mb,
    get_gpu_vram_mb,
    get_model_size_mb,
    calculate_weighted_overall_score,
    EnergyBenchmarkTracker,
)

logger = get_logger("BenchmarkDatasetRunner")

DEFAULT_RESULTS_CSV = CFG_ROOT / "results" / "results.csv"
DEFAULT_MANIFEST_PATH = CFG_ROOT / "dataset" / "manifest.csv"

# Edge Models to evaluate for STT benchmark matrix
BENCHMARK_MODELS = [
    "indicconformer",
    "whisper_tiny",
    "whisper_small",
]

TARGET_LANGUAGES = ["hi", "ta", "te"]


@dataclass
class EvaluationSummary:
    model_name: str
    model_version: str
    runtime: str
    precision: str
    language: str
    dataset_name: str
    split: str
    noise_condition: str
    num_samples: int
    num_threads: int
    device: str
    wer: Optional[float]
    cer: Optional[float]
    latency_cold_sec: float
    latency_warm_sec: float
    rtf: float
    ram_mb: float
    vram_mb: Optional[float]
    model_size_mb: float
    score: Optional[float]
    timestamp: str
    run_id: str
    status: str = "COMPLETED"
    notes: str = ""


class BenchmarkDatasetRunner:
    """
    Decoupled benchmark execution runner across public dataset manifests.
    """

    def __init__(
        self,
        app_config: Optional[AppConfig] = None,
        manifest_path: Path = DEFAULT_MANIFEST_PATH,
        results_csv_path: Path = DEFAULT_RESULTS_CSV,
    ):
        self.app_config = app_config or AppConfig.load()
        self.manifest_path = Path(manifest_path)
        self.results_csv_path = Path(results_csv_path)
        # Benchmark runner uses benchmark_mode=True to allow warm latency measurement
        self.model_manager = ModelManager(self.app_config, benchmark_mode=True)

    def load_manifest(self) -> pd.DataFrame:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path.resolve()}")
        df = pd.read_csv(self.manifest_path, encoding="utf-8")
        return df

    def warmup_and_cache_models(self, models: List[str], languages: List[str]) -> None:
        """
        Pre-downloads and verifies all required models in local disk cache before timing starts.
        Ensures cold latency measures disk-to-memory initialization only, never network download.
        """
        logger.info("=" * 60)
        logger.info("Pre-warming and verifying local disk cache for all models...")
        logger.info("=" * 60)

        for model_name in models:
            for lang in languages:
                try:
                    logger.info(f"--> Pre-caching [{model_name}] for [{lang.upper()}]...")
                    engine = self.model_manager.load_stt(lang, model_name=model_name)
                    assert engine.is_loaded()
                    logger.info(f"    [OK] Cached: {model_name} ({lang})")
                except Exception as e:
                    logger.warning(f"    [WARN] Failed during pre-cache of {model_name} ({lang}): {e}")
                finally:
                    self.model_manager.unload_all()
                    import gc
                    gc.collect()

        logger.info("Pre-caching complete. All model weights verified on local disk.\n")

    def run_benchmark(
        self,
        models: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        conditions: Optional[List[str]] = None,
        run_id: Optional[str] = None,
        pre_warmup: bool = True,
    ) -> List[EvaluationSummary]:
        df = self.load_manifest()
        eval_models = models or BENCHMARK_MODELS
        eval_langs = languages or TARGET_LANGUAGES
        eval_conditions = conditions or ["clean", "noisy"]
        current_run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        logger.info(
            f"Starting benchmark run [{current_run_id}] | Models: {eval_models} | "
            f"Langs: {eval_langs} | Conditions: {eval_conditions}"
        )

        # Pre-cache all models before timed matrix execution
        if pre_warmup:
            self.warmup_and_cache_models(eval_models, eval_langs)

        summaries: List[EvaluationSummary] = []

        for model_name in eval_models:
            for lang in eval_langs:
                for cond in eval_conditions:
                    # Filter manifest subset
                    subset = df[(df["language"] == lang) & (df["noise_condition"] == cond)]

                    if subset.empty:
                        logger.warning(f"No samples found for ({model_name}, {lang}, {cond}). Recording UNAVAILABLE.")
                        summary = EvaluationSummary(
                            model_name=model_name,
                            model_version="N/A",
                            runtime="N/A",
                            precision="N/A",
                            language=lang,
                            dataset_name="kathbath",
                            split="test",
                            noise_condition=cond,
                            num_samples=0,
                            num_threads=self.app_config.num_threads,
                            device="cpu",
                            wer=None,
                            cer=None,
                            latency_cold_sec=0.0,
                            latency_warm_sec=0.0,
                            rtf=0.0,
                            ram_mb=0.0,
                            vram_mb=None,
                            model_size_mb=0.0,
                            score=None,
                            timestamp=timestamp,
                            run_id=current_run_id,
                            status="UNAVAILABLE",
                            notes=f"No dataset split available in manifest for {lang} [{cond}]",
                        )
                        summaries.append(summary)
                        self._append_to_csv([summary])
                        continue

                    # Evaluate cell
                    summary = self._evaluate_cell(
                        model_name=model_name,
                        language=lang,
                        condition=cond,
                        subset=subset,
                        run_id=current_run_id,
                        timestamp=timestamp,
                    )
                    summaries.append(summary)
                    self._append_to_csv([summary])

        # Clean up models after benchmark run
        self.model_manager.unload_all()
        logger.info(f"Benchmark run [{current_run_id}] finished. Output saved to {self.results_csv_path}")
        return summaries

    def _evaluate_cell(
        self,
        model_name: str,
        language: str,
        condition: str,
        subset: pd.DataFrame,
        run_id: str,
        timestamp: str,
    ) -> EvaluationSummary:
        logger.info(f"--> Evaluating [{model_name}] on [{language.upper()}] [{condition}] ({len(subset)} samples)...")

        # Unload previously active model to measure true cold disk-to-RAM load time
        self.model_manager.unload_all()

        try:
            cfg = self.model_manager.registry.get_stt_config(language, model_name=model_name)
        except Exception as e:
            logger.error(f"Configuration not found for ({model_name}, {language}): {e}")
            return EvaluationSummary(
                model_name=model_name,
                model_version="unknown",
                runtime="unknown",
                precision="unknown",
                language=language,
                dataset_name="kathbath",
                split="test",
                noise_condition=condition,
                num_samples=len(subset),
                num_threads=self.app_config.num_threads,
                device="cpu",
                wer=None,
                cer=None,
                latency_cold_sec=0.0,
                latency_warm_sec=0.0,
                rtf=0.0,
                ram_mb=0.0,
                vram_mb=None,
                model_size_mb=0.0,
                score=None,
                timestamp=timestamp,
                run_id=run_id,
                status="ERROR",
                notes=str(e),
            )

        model_path = cfg.get_absolute_model_path()
        model_size_mb = get_model_size_mb(model_path)

        # 1. Cold Load (from local cache on disk into RAM)
        load_start = time.perf_counter()
        try:
            engine = self.model_manager.load_stt(language, model_name=model_name)
            disk_load_time = time.perf_counter() - load_start
        except Exception as e:
            logger.error(f"Failed to load model {model_name} for {language}: {e}")
            return EvaluationSummary(
                model_name=model_name,
                model_version=cfg.version,
                runtime=cfg.runtime,
                precision=cfg.quantization,
                language=language,
                dataset_name="kathbath",
                split="test",
                noise_condition=condition,
                num_samples=len(subset),
                num_threads=self.app_config.num_threads,
                device=cfg.device,
                wer=None,
                cer=None,
                latency_cold_sec=0.0,
                latency_warm_sec=0.0,
                rtf=0.0,
                ram_mb=get_process_rss_mb(),
                vram_mb=get_gpu_vram_mb(),
                model_size_mb=model_size_mb,
                score=None,
                timestamp=timestamp,
                run_id=run_id,
                status="LOAD_FAILED",
                notes=str(e),
            )

        total_audio_duration = 0.0
        total_inference_time = 0.0
        sample_wers = []
        sample_cers = []
        peak_ram = get_process_rss_mb()
        cold_latency = 0.0
        warm_latencies = []

        first_sample = True
        for _, row in subset.iterrows():
            audio_path_raw = row["audio_path"]
            audio_path = Path(audio_path_raw)
            if not audio_path.is_absolute():
                audio_path = CFG_ROOT / audio_path

            if not audio_path.exists():
                logger.warning(f"Audio file missing: {audio_path}")
                continue

            try:
                audio = AudioInput.from_wav_file(audio_path)
            except Exception as e:
                logger.warning(f"Failed to load audio {audio_path}: {e}")
                continue

            start_t = time.perf_counter()
            result = engine.transcribe(audio)
            inf_t = time.perf_counter() - start_t

            if first_sample:
                cold_latency = disk_load_time + inf_t
                first_sample = False

            warm_latencies.append(inf_t)
            total_audio_duration += audio.duration_sec
            total_inference_time += inf_t

            # Track peak process RAM
            curr_ram = get_process_rss_mb()
            if curr_ram > peak_ram:
                peak_ram = curr_ram

            # Accuracy metrics
            ref_text = str(row.get("reference_text", "")).strip()
            if ref_text and ref_text != "nan":
                wer, cer = compute_accuracy_metrics(ref_text, result.text)
                if wer is not None:
                    sample_wers.append(wer)
                if cer is not None:
                    sample_cers.append(cer)

        avg_wer = float(sum(sample_wers) / len(sample_wers)) if sample_wers else None
        avg_cer = float(sum(sample_cers) / len(sample_cers)) if sample_cers else None
        avg_warm_lat = float(sum(warm_latencies) / len(warm_latencies)) if warm_latencies else 0.0
        overall_rtf = total_inference_time / total_audio_duration if total_audio_duration > 0 else 0.0
        vram_mb = get_gpu_vram_mb()

        # Weighted score (0 to 100) based on accuracy, latency, RAM, and model size
        score = calculate_weighted_overall_score(
            wer=avg_wer,
            latency_warm_sec=avg_warm_lat,
            ram_mb=peak_ram,
            model_size_mb=model_size_mb,
        )

        return EvaluationSummary(
            model_name=model_name,
            model_version=cfg.version,
            runtime=cfg.runtime,
            precision=cfg.quantization,
            language=language,
            dataset_name=str(subset.iloc[0].get("dataset_name", "kathbath")),
            split=str(subset.iloc[0].get("split", "test")),
            noise_condition=condition,
            num_samples=len(warm_latencies),
            num_threads=self.app_config.num_threads,
            device=cfg.device,
            wer=round(avg_wer, 4) if avg_wer is not None else None,
            cer=round(avg_cer, 4) if avg_cer is not None else None,
            latency_cold_sec=round(cold_latency, 3),
            latency_warm_sec=round(avg_warm_lat, 3),
            rtf=round(overall_rtf, 4),
            ram_mb=round(peak_ram, 1),
            vram_mb=round(vram_mb, 1) if vram_mb is not None else None,
            model_size_mb=round(model_size_mb, 1),
            score=score,
            timestamp=timestamp,
            run_id=run_id,
            status="COMPLETED",
        )

    def _append_to_csv(self, summaries: List[EvaluationSummary]) -> None:
        self.results_csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.results_csv_path.exists() and self.results_csv_path.stat().st_size > 0

        fieldnames = [
            "run_id",
            "timestamp",
            "model_name",
            "model_version",
            "runtime",
            "precision",
            "language",
            "dataset_name",
            "split",
            "noise_condition",
            "num_samples",
            "num_threads",
            "device",
            "wer",
            "cer",
            "latency_cold_sec",
            "latency_warm_sec",
            "rtf",
            "ram_mb",
            "vram_mb",
            "model_size_mb",
            "score",
            "status",
            "notes",
        ]

        with open(self.results_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            for s in summaries:
                writer.writerow({
                    "run_id": s.run_id,
                    "timestamp": s.timestamp,
                    "model_name": s.model_name,
                    "model_version": s.model_version,
                    "runtime": s.runtime,
                    "precision": s.precision,
                    "language": s.language,
                    "dataset_name": s.dataset_name,
                    "split": s.split,
                    "noise_condition": s.noise_condition,
                    "num_samples": s.num_samples,
                    "num_threads": s.num_threads,
                    "device": s.device,
                    "wer": f"{s.wer:.4f}" if s.wer is not None else "",
                    "cer": f"{s.cer:.4f}" if s.cer is not None else "",
                    "latency_cold_sec": f"{s.latency_cold_sec:.3f}",
                    "latency_warm_sec": f"{s.latency_warm_sec:.3f}",
                    "rtf": f"{s.rtf:.4f}",
                    "ram_mb": f"{s.ram_mb:.1f}",
                    "vram_mb": f"{s.vram_mb:.1f}" if s.vram_mb is not None else "",
                    "model_size_mb": f"{s.model_size_mb:.1f}",
                    "score": f"{s.score:.2f}" if s.score is not None else "",
                    "status": s.status,
                    "notes": s.notes,
                })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run dataset benchmark.")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST_PATH), help="Manifest CSV path.")
    parser.add_argument("--results", type=str, default=str(DEFAULT_RESULTS_CSV), help="Results CSV output path.")
    parser.add_argument("--models", nargs="+", default=None, help="Specific models to benchmark.")
    parser.add_argument("--languages", nargs="+", default=None, help="Specific languages (hi, ta, te).")
    parser.add_argument("--conditions", nargs="+", default=None, help="Conditions (clean, noisy).")
    args = parser.parse_args()

    runner = BenchmarkDatasetRunner(
        manifest_path=Path(args.manifest),
        results_csv_path=Path(args.results),
    )
    runner.run_benchmark(
        models=args.models,
        languages=args.languages,
        conditions=args.conditions,
    )
