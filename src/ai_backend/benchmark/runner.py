"""
Decoupled Benchmark Runner.
Executes batch or single-file STT evaluation, computes metrics, and appends to CSV results log.
"""

import csv
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from ai_backend.core.types import AudioInput, TranscriptionResult
from ai_backend.pipeline.speech_pipeline import SpeechPipeline
from ai_backend.benchmark.metrics import compute_accuracy_metrics


@dataclass
class BenchmarkRecord:
    filename: str
    language: str
    audio_duration_sec: float
    inference_time_sec: float
    rtf: float
    transcription: str
    reference: Optional[str]
    wer: Optional[float]
    cer: Optional[float]
    model_name: str
    quantization: str


class BenchmarkRunner:
    """
    Decoupled benchmark execution runner.
    """

    def __init__(self, pipeline: Optional[SpeechPipeline] = None):
        self.pipeline = pipeline or SpeechPipeline()

    def find_reference_text(self, audio_path: Path) -> Optional[str]:
        candidates = [
            audio_path.with_suffix(".txt"),
            audio_path.with_suffix(".ref.txt"),
            audio_path.parent / f"{audio_path.stem}.txt",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                try:
                    return candidate.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
        return None

    def run_file(
        self,
        audio_path: Path,
        language: str = "hi",
        reference_text: Optional[str] = None
    ) -> BenchmarkRecord:
        audio = AudioInput.from_wav_file(audio_path)
        result = self.pipeline.transcribe(audio, language=language)

        ref = reference_text if reference_text is not None else self.find_reference_text(audio_path)
        wer, cer = compute_accuracy_metrics(ref, result.text)

        return BenchmarkRecord(
            filename=audio_path.name,
            language=language,
            audio_duration_sec=result.audio_duration_sec,
            inference_time_sec=result.inference_time_sec,
            rtf=result.rtf,
            transcription=result.text,
            reference=ref,
            wer=wer,
            cer=cer,
            model_name=result.model_name,
            quantization=result.quantization,
        )

    def save_results_to_csv(self, records: List[BenchmarkRecord], csv_path: Path) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = csv_path.exists() and csv_path.stat().st_size > 0

        fieldnames = [
            "filename",
            "language",
            "audio_duration_sec",
            "inference_time_sec",
            "rtf",
            "transcription",
            "wer",
            "cer",
        ]

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            for rec in records:
                writer.writerow({
                    "filename": rec.filename,
                    "language": rec.language,
                    "audio_duration_sec": f"{rec.audio_duration_sec:.2f}",
                    "inference_time_sec": f"{rec.inference_time_sec:.2f}",
                    "rtf": f"{rec.rtf:.2f}",
                    "transcription": rec.transcription,
                    "wer": f"{rec.wer:.4f}" if rec.wer is not None else "",
                    "cer": f"{rec.cer:.4f}" if rec.cer is not None else "",
                })
