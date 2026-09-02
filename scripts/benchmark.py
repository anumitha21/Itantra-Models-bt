"""
CLI Script to benchmark STT models using AI Backend BenchmarkRunner.
Usage:
    python scripts/benchmark.py --language hi --input test_audio/hindi/
"""

import sys
import argparse
from pathlib import Path

# Ensure src/ is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ai_backend.benchmark.runner import BenchmarkRunner, BenchmarkRecord
from ai_backend.core.config import AppConfig


def print_record(rec: BenchmarkRecord):
    print("-" * 40)
    print(f"Language: {rec.language}")
    print(f"Model: {rec.model_name} ({rec.quantization})")
    print(f"Audio duration: {rec.audio_duration_sec:.2f} sec")
    print(f"Inference time: {rec.inference_time_sec:.2f} sec")
    print(f"RTF: {rec.rtf:.2f}")
    if rec.wer is not None:
        print(f"WER: {rec.wer * 100:.2f}% | CER: {rec.cer * 100:.2f}%")
    print("\nTranscription:")
    print(rec.transcription)
    print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="Run STT Benchmark.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to file or folder containing WAV files.")
    parser.add_argument("--language", "-l", type=str, default="hi", help="Language code (default: hi).")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output CSV path.")

    args = parser.parse_args()
    input_path = Path(args.input)
    app_cfg = AppConfig.load()
    csv_path = Path(args.output) if args.output else app_cfg.results_csv

    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    audio_files = []
    if input_path.is_file():
        audio_files.append(input_path)
    elif input_path.is_dir():
        audio_files = sorted(list(input_path.glob("**/*.wav")))

    if not audio_files:
        print(f"Error: No .wav files found in '{input_path}'.", file=sys.stderr)
        sys.exit(1)

    runner = BenchmarkRunner()
    records = []

    print(f"\nRunning Offline STT Benchmark on {len(audio_files)} file(s)...")

    for audio_file in audio_files:
        try:
            rec = runner.run_file(audio_file, language=args.language)
            print_record(rec)
            records.append(rec)
        except Exception as e:
            print(f"[ERROR] Benchmark failed for '{audio_file.name}': {e}", file=sys.stderr)

    if records:
        runner.save_results_to_csv(records, csv_path)
        print(f"\nResults successfully saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
