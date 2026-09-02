"""
TTS Benchmark Dataset Runner for Reproducible Multi-Model TTS Evaluation.
Evaluates AI4Bharat Indic-TTS and Meta MMS-TTS across Hindi, Tamil, and Telugu.
Measures:
  1. Standalone TTS metrics: synthesis latency, synthesized audio duration, RTF, standalone RAM (ram_mb_tts_only), model size.
  2. STT Judge metrics: combined RAM (ram_mb_roundtrip_combined), Round-Trip WER/CER, STT Judge Baseline WER, and TTS-Attributable WER.
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
sys.path.insert(0, str(PROJECT_ROOT))

from ai_backend.core.config import AppConfig, PROJECT_ROOT as CFG_ROOT
from ai_backend.core.types import AudioInput
from ai_backend.core.logging import get_logger
from ai_backend.models.model_manager import ModelManager
from ai_backend.benchmark.metrics import (
    compute_accuracy_metrics,
    get_process_rss_mb,
    get_gpu_vram_mb,
    get_model_size_mb,
    EnergyBenchmarkTracker,
)
from benchmark.dataset_runner import DEFAULT_RESULTS_CSV, DEFAULT_MANIFEST_PATH

logger = get_logger("TTSBenchmarkDatasetRunner")

DEFAULT_TTS_RESULTS_CSV = CFG_ROOT / "results" / "tts_results.csv"

# TTS models and target languages
TTS_BENCHMARK_MODELS = [
    "ai4bharat_vits",
    "mms_vits",
]

TTS_TARGET_LANGUAGES = ["hi", "ta", "te"]


@dataclass
class TTSEvaluationSummary:
    run_id: str
    timestamp: str
    tts_model_name: str
    tts_version: str
    runtime: str
    precision: str
    language: str
    input_text_source: str
    num_samples: int
    num_threads: int
    device: str
    tts_native_sample_rate: int
    synthesized_audio_duration_sec: float
    roundtrip_wer: Optional[float]
    roundtrip_cer: Optional[float]
    stt_judge_baseline_wer: Optional[float]
    tts_attributable_wer: Optional[float]
    synthesis_latency_sec: float
    rtf: float
    ram_mb_tts_only: float
    ram_mb_roundtrip_combined: float
    model_size_mb: float
    status: str = "COMPLETED"
    notes: str = ""


class TTSBenchmarkDatasetRunner:
    """
    Runner for offline TTS benchmark evaluation with STT round-trip intelligibility testing.
    """

    def __init__(
        self,
        app_config: Optional[AppConfig] = None,
        manifest_path: Path = DEFAULT_MANIFEST_PATH,
        tts_results_csv_path: Path = DEFAULT_TTS_RESULTS_CSV,
        stt_results_csv_path: Path = DEFAULT_RESULTS_CSV,
    ):
        self.app_config = app_config or AppConfig.load()
        self.manifest_path = Path(manifest_path)
        self.tts_results_csv_path = Path(tts_results_csv_path)
        self.stt_results_csv_path = Path(stt_results_csv_path)
        self._stt_baseline_wers = self._load_stt_baseline_wers()

    def _load_stt_baseline_wers(self) -> Dict[str, float]:
        """
        Extract IndicConformer clean dictation WER baseline per language from STT results.csv.
        """
        baselines = {}
        for path in [self.stt_results_csv_path, CFG_ROOT / "results" / "r1.csv"]:
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    ic_df = df[(df["model_name"] == "indicconformer") & (df["noise_condition"] == "clean")]
                    for _, row in ic_df.iterrows():
                        lang = str(row["language"]).lower().strip()
                        wer_val = float(row["wer"])
                        if lang not in baselines and not pd.isna(wer_val):
                            baselines[lang] = wer_val
                except Exception as e:
                    logger.debug(f"Could not load STT baselines from {path}: {e}")

        # Default fallback baselines if results.csv not yet generated
        default_baselines = {"hi": 0.1109, "ta": 0.2862, "te": 0.2051}
        for k, v in default_baselines.items():
            if k not in baselines:
                baselines[k] = v
        return baselines

    def load_manifest_sentences(self, language: str, max_samples: Optional[int] = None) -> List[str]:
        """
        Extract real reference sentences from Kathbath dataset manifest and transcripts for a target language.
        """
        sentences = []

        # 1. Load from manifest.csv (clean and noisy)
        if self.manifest_path.exists():
            try:
                df = pd.read_csv(self.manifest_path, encoding="utf-8")
                subset = df[df["language"] == language]
                for text in subset["reference_text"].dropna():
                    t = str(text).strip()
                    if t and t != "nan" and t not in sentences:
                        sentences.append(t)
            except Exception as e:
                logger.debug(f"Manifest read error: {e}")

        # 2. If more sentences needed, draw from official Kathbath clean test_known transcripts
        target_count = max_samples if (max_samples and max_samples > 0) else 30
        if len(sentences) < target_count:
            lang_dir_names = {"hi": "hindi", "ta": "tamil", "te": "telugu"}
            folder = lang_dir_names.get(language, language)
            trans_path = CFG_ROOT / "data" / "kathbath" / "clean" / "transcripts" / "kb_data_clean_m4a" / folder / "test_known" / "transcription_n2w.txt"
            if trans_path.exists():
                try:
                    for line in trans_path.read_text(encoding="utf-8").strip().splitlines():
                        parts = line.strip().split(None, 1)
                        if len(parts) > 1:
                            t = parts[1].strip()
                            if t and t not in sentences:
                                sentences.append(t)
                                if len(sentences) >= target_count:
                                    break
                except Exception as e:
                    logger.debug(f"Kathbath transcript read error: {e}")

        if max_samples and max_samples > 0:
            sentences = sentences[:max_samples]

        return sentences

    def run_benchmark(
        self,
        models: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        max_samples_per_cell: Optional[int] = 10,
        run_id: Optional[str] = None,
        overwrite: bool = True,
    ) -> List[TTSEvaluationSummary]:
        eval_models = models or TTS_BENCHMARK_MODELS
        eval_langs = languages or TTS_TARGET_LANGUAGES
        current_run_id = run_id or f"tts_run_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        logger.info(
            f"Starting TTS Benchmark Run [{current_run_id}] | Models: {eval_models} | "
            f"Langs: {eval_langs} | Max Samples/Cell: {max_samples_per_cell}"
        )

        if overwrite and self.tts_results_csv_path.exists():
            try:
                self.tts_results_csv_path.unlink()
            except Exception as e:
                logger.debug(f"Could not remove old tts_results.csv: {e}")

        summaries: List[TTSEvaluationSummary] = []

        for model_name in eval_models:
            for lang in eval_langs:
                sentences = self.load_manifest_sentences(lang, max_samples=max_samples_per_cell)

                if not sentences:
                    logger.warning(f"No sentences found for ({model_name}, {lang}). Recording UNAVAILABLE.")
                    stt_base = self._stt_baseline_wers.get(lang)
                    summary = TTSEvaluationSummary(
                        run_id=current_run_id,
                        timestamp=timestamp,
                        tts_model_name=model_name,
                        tts_version="N/A",
                        runtime="sherpa-onnx",
                        precision="fp32",
                        language=lang,
                        input_text_source="kathbath",
                        num_samples=0,
                        num_threads=self.app_config.num_threads,
                        device="cpu",
                        tts_native_sample_rate=0,
                        synthesized_audio_duration_sec=0.0,
                        roundtrip_wer=None,
                        roundtrip_cer=None,
                        stt_judge_baseline_wer=stt_base,
                        tts_attributable_wer=None,
                        synthesis_latency_sec=0.0,
                        rtf=0.0,
                        ram_mb_tts_only=0.0,
                        ram_mb_roundtrip_combined=0.0,
                        model_size_mb=0.0,
                        status="UNAVAILABLE",
                        notes=f"No reference sentences in manifest for {lang}",
                    )
                    summaries.append(summary)
                    self._append_to_csv([summary])
                    continue

                summary = self._evaluate_cell(
                    model_name=model_name,
                    language=lang,
                    sentences=sentences,
                    run_id=current_run_id,
                    timestamp=timestamp,
                )
                summaries.append(summary)
                self._append_to_csv([summary])

        logger.info(f"TTS Benchmark run [{current_run_id}] finished. Output saved to {self.tts_results_csv_path}")
        return summaries

    def _evaluate_cell(
        self,
        model_name: str,
        language: str,
        sentences: List[str],
        run_id: str,
        timestamp: str,
    ) -> TTSEvaluationSummary:
        logger.info(f"--> Evaluating TTS [{model_name}] on [{language.upper()}] ({len(sentences)} sentences)...")

        # -------------------------------------------------------------
        # PASS 1: Standalone TTS Inference & TTS-Only Peak RAM
        # -------------------------------------------------------------
        mgr_tts_only = ModelManager(self.app_config, benchmark_mode=False)
        mgr_tts_only.unload_all()

        try:
            tts_cfg = mgr_tts_only.registry.get_tts_config(language, model_name=model_name)
        except Exception as e:
            logger.error(f"TTS Configuration not found for ({model_name}, {language}): {e}")
            stt_base = self._stt_baseline_wers.get(language)
            return TTSEvaluationSummary(
                run_id=run_id,
                timestamp=timestamp,
                tts_model_name=model_name,
                tts_version="unknown",
                runtime="sherpa-onnx",
                precision="unknown",
                language=language,
                input_text_source="kathbath",
                num_samples=len(sentences),
                num_threads=self.app_config.num_threads,
                device="cpu",
                tts_native_sample_rate=0,
                synthesized_audio_duration_sec=0.0,
                roundtrip_wer=None,
                roundtrip_cer=None,
                stt_judge_baseline_wer=stt_base,
                tts_attributable_wer=None,
                synthesis_latency_sec=0.0,
                rtf=0.0,
                ram_mb_tts_only=0.0,
                ram_mb_roundtrip_combined=0.0,
                model_size_mb=0.0,
                status="CONFIG_ERROR",
                notes=str(e),
            )

        model_path = tts_cfg.get_absolute_model_path()
        model_size_mb = get_model_size_mb(model_path)

        try:
            tts_engine = mgr_tts_only.load_tts(language, model_name=model_name)
        except Exception as e:
            logger.error(f"Failed to load standalone TTS engine for ({model_name}, {language}): {e}")
            stt_base = self._stt_baseline_wers.get(language)
            return TTSEvaluationSummary(
                run_id=run_id,
                timestamp=timestamp,
                tts_model_name=model_name,
                tts_version=tts_cfg.version,
                runtime=tts_cfg.runtime,
                precision=tts_cfg.quantization,
                language=language,
                input_text_source="kathbath",
                num_samples=len(sentences),
                num_threads=self.app_config.num_threads,
                device=tts_cfg.device,
                tts_native_sample_rate=0,
                synthesized_audio_duration_sec=0.0,
                roundtrip_wer=None,
                roundtrip_cer=None,
                stt_judge_baseline_wer=stt_base,
                tts_attributable_wer=None,
                synthesis_latency_sec=0.0,
                rtf=0.0,
                ram_mb_tts_only=get_process_rss_mb(),
                ram_mb_roundtrip_combined=0.0,
                model_size_mb=model_size_mb,
                status="LOAD_FAILED",
                notes=str(e),
            )

        synth_latencies = []
        synth_audios: List[AudioInput] = []
        valid_sentences: List[str] = []
        total_audio_duration = 0.0
        total_synth_latency = 0.0
        peak_ram_tts_only = get_process_rss_mb()
        native_sample_rate = tts_cfg.expected_sample_rate

        for idx, text in enumerate(sentences):
            start_t = time.perf_counter()
            try:
                audio = tts_engine.synthesize(text, language=language)
                lat = time.perf_counter() - start_t
            except Exception as e:
                logger.warning(f"Synthesis failed for sample {idx+1} '{text[:30]}...': {e}")
                continue

            synth_latencies.append(lat)
            synth_audios.append(audio)
            valid_sentences.append(text)
            total_synth_latency += lat
            total_audio_duration += audio.duration_sec
            native_sample_rate = audio.sample_rate

            curr_ram = get_process_rss_mb()
            if curr_ram > peak_ram_tts_only:
                peak_ram_tts_only = curr_ram

        # Unload standalone TTS
        mgr_tts_only.unload_all()

        # -------------------------------------------------------------
        # PASS 2: Combined STT Judge Round-Trip Evaluation & Combined RAM
        # -------------------------------------------------------------
        mgr_combined = ModelManager(self.app_config, benchmark_mode=True)
        mgr_combined.unload_all()

        peak_ram_combined = get_process_rss_mb()
        sample_wers = []
        sample_cers = []

        try:
            judge_stt = mgr_combined.load_stt(language, model_name="indicconformer")
            judge_ram = get_process_rss_mb()
            if judge_ram > peak_ram_combined:
                peak_ram_combined = judge_ram

            judge_tts = mgr_combined.load_tts(language, model_name=model_name)
            combined_ram = get_process_rss_mb()
            if combined_ram > peak_ram_combined:
                peak_ram_combined = combined_ram

            for audio, ref_text in zip(synth_audios, valid_sentences):
                audio_16k = audio.resample(16000)
                stt_res = judge_stt.transcribe(audio_16k)
                curr_ram = get_process_rss_mb()
                if curr_ram > peak_ram_combined:
                    peak_ram_combined = curr_ram

                wer, cer = compute_accuracy_metrics(ref_text, stt_res.text)
                if wer is not None:
                    sample_wers.append(wer)
                if cer is not None:
                    sample_cers.append(cer)

        except Exception as e:
            logger.error(f"Combined STT Judge round-trip evaluation error: {e}")
        finally:
            mgr_combined.unload_all()

        avg_wer = float(sum(sample_wers) / len(sample_wers)) if sample_wers else None
        avg_cer = float(sum(sample_cers) / len(sample_cers)) if sample_cers else None
        avg_synth_lat = float(sum(synth_latencies) / len(synth_latencies)) if synth_latencies else 0.0
        avg_duration = float(total_audio_duration / len(synth_audios)) if synth_audios else 0.0
        overall_rtf = total_synth_latency / total_audio_duration if total_audio_duration > 0 else 0.0

        stt_baseline = self._stt_baseline_wers.get(language)
        tts_attributable_wer = (avg_wer - stt_baseline) if (avg_wer is not None and stt_baseline is not None) else None

        return TTSEvaluationSummary(
            run_id=run_id,
            timestamp=timestamp,
            tts_model_name=model_name,
            tts_version=tts_cfg.version,
            runtime=tts_cfg.runtime,
            precision=tts_cfg.quantization,
            language=language,
            input_text_source="kathbath",
            num_samples=len(valid_sentences),
            num_threads=self.app_config.num_threads,
            device=tts_cfg.device,
            tts_native_sample_rate=native_sample_rate,
            synthesized_audio_duration_sec=round(avg_duration, 4),
            roundtrip_wer=round(avg_wer, 4) if avg_wer is not None else None,
            roundtrip_cer=round(avg_cer, 4) if avg_cer is not None else None,
            stt_judge_baseline_wer=round(stt_baseline, 4) if stt_baseline is not None else None,
            tts_attributable_wer=round(tts_attributable_wer, 4) if tts_attributable_wer is not None else None,
            synthesis_latency_sec=round(avg_synth_lat, 4),
            rtf=round(overall_rtf, 4),
            ram_mb_tts_only=round(peak_ram_tts_only, 1),
            ram_mb_roundtrip_combined=round(peak_ram_combined, 1),
            model_size_mb=round(model_size_mb, 1),
            status="COMPLETED",
            notes="Round-trip intelligibility via IndicConformer (16kHz)",
        )

    def _append_to_csv(self, summaries: List[TTSEvaluationSummary]) -> None:
        self.tts_results_csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.tts_results_csv_path.exists()

        fieldnames = [
            "run_id",
            "timestamp",
            "tts_model_name",
            "tts_version",
            "runtime",
            "precision",
            "language",
            "input_text_source",
            "num_samples",
            "num_threads",
            "device",
            "tts_native_sample_rate",
            "synthesized_audio_duration_sec",
            "roundtrip_wer",
            "roundtrip_cer",
            "stt_judge_baseline_wer",
            "tts_attributable_wer",
            "synthesis_latency_sec",
            "rtf",
            "ram_mb_tts_only",
            "ram_mb_roundtrip_combined",
            "model_size_mb",
            "status",
            "notes",
        ]

        with open(self.tts_results_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists or os.path.getsize(self.tts_results_csv_path) == 0:
                writer.writeheader()
            for s in summaries:
                writer.writerow(s.__dict__)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Offline TTS Intelligibility Benchmark.")
    parser.add_argument("--models", nargs="+", default=["ai4bharat_vits", "mms_vits"], help="TTS Models to evaluate.")
    parser.add_argument("--languages", nargs="+", default=["hi", "ta", "te"], help="Languages to evaluate (hi, ta, te).")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of sentences per language cell.")
    args = parser.parse_args()

    runner = TTSBenchmarkDatasetRunner()
    results = runner.run_benchmark(
        models=args.models,
        languages=args.languages,
        max_samples_per_cell=args.num_samples,
        overwrite=True,
    )
    print("\n" + "=" * 100)
    print("TTS BENCHMARK EVALUATION COMPLETE")
    print("=" * 100)
    for r in results:
        print(f"[{r.tts_model_name}] {r.language.upper()} | Native SR: {r.tts_native_sample_rate}Hz | "
              f"Roundtrip WER: {r.roundtrip_wer:.4f} (STT Base: {r.stt_judge_baseline_wer:.4f}, Attrib: {r.tts_attributable_wer:.4f}) | "
              f"Latency: {r.synthesis_latency_sec:.2f}s, Dur: {r.synthesized_audio_duration_sec:.2f}s, RTF: {r.rtf:.4f} | "
              f"RAM (TTS-only): {r.ram_mb_tts_only}MB, RAM (Combined): {r.ram_mb_roundtrip_combined}MB")
