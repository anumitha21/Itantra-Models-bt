"""
Offline STT Benchmark Tool for AI4Bharat IndicConformer ONNX Baseline.

Usage:
    python benchmark/benchmark_stt.py test_audio/hindi/test01.wav
    python benchmark/benchmark_stt.py test_audio/english/
    python benchmark/benchmark_stt.py test_audio/hindi/test01.wav --lang hindi --num-threads 2
"""

import sys
import os
import csv
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

# Ensure parent directory is in sys.path for src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding for Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


from src.config import STTConfig, SUPPORTED_LANGUAGES, DEFAULT_RESULTS_CSV
from src.audio import load_audio, AudioProcessingError
from src.transcriber import IndicConformerTranscriber, TranscriberError
from src.metrics import compute_metrics


def detect_language(path: Path, user_lang: Optional[str] = None) -> str:
    """
    Detect target language from user parameter or file/directory path.
    """
    if user_lang:
        lang_key = user_lang.lower().strip()
        if lang_key in SUPPORTED_LANGUAGES:
            return SUPPORTED_LANGUAGES[lang_key]
        raise ValueError(
            f"Unsupported language '{user_lang}'. Supported choices: hindi, english."
        )

    path_str = str(path).lower()
    if "english" in path_str or "\\english" in path_str or "/english" in path_str or "_en" in path_str:
        return "en"
    if "hindi" in path_str or "\\hindi" in path_str or "/hindi" in path_str or "_hi" in path_str:
        return "hi"

    # Default fallback to Hindi if ambiguous
    return "hi"


def find_reference_text(audio_path: Path) -> Optional[str]:
    """
    Look for a reference ground-truth text file with matching stem (e.g. test01.txt).
    """
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


def save_to_csv(csv_path: Path, records: List[Dict[str, Any]]) -> None:
    """
    Append benchmark records to the CSV file. Writes header if file is new.
    """
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
                "filename": rec["filename"],
                "language": rec["language"],
                "audio_duration_sec": f"{rec['audio_duration_sec']:.2f}",
                "inference_time_sec": f"{rec['inference_time_sec']:.2f}",
                "rtf": f"{rec['rtf']:.2f}",
                "transcription": rec["transcription"],
                "wer": f"{rec['wer']:.4f}" if rec["wer"] is not None else "",
                "cer": f"{rec['cer']:.4f}" if rec["cer"] is not None else "",
            })


def process_single_file(
    audio_path: Path,
    transcribers: Dict[str, IndicConformerTranscriber],
    user_lang: Optional[str] = None,
    config: Optional[STTConfig] = None
) -> Dict[str, Any]:
    """
    Process a single WAV audio file and return benchmark metrics.
    """
    lang_code = detect_language(audio_path, user_lang)
    lang_name = "Hindi" if lang_code == "hi" else "English"

    # Reuse transcriber for the detected language
    if lang_code not in transcribers:
        stt_config = config or STTConfig(language=lang_code)
        stt_config.language = lang_code
        transcribers[lang_code] = IndicConformerTranscriber(config=stt_config)

    transcriber = transcribers[lang_code]

    # Load & resample audio
    samples, duration_sec, sr = load_audio(audio_path, target_sample_rate=16000)

    # Perform offline transcription
    result = transcriber.transcribe(samples, sample_rate=sr)
    transcription = result["transcription"]
    inference_time_sec = result["inference_time_sec"]

    # Real-Time Factor (RTF = inference_time / audio_duration)
    rtf = inference_time_sec / duration_sec if duration_sec > 0 else 0.0

    # Optional reference text & accuracy computation
    ref_text = find_reference_text(audio_path)
    wer, cer = compute_metrics(ref_text, transcription)

    record = {
        "filename": audio_path.name,
        "full_path": str(audio_path.resolve()),
        "language": lang_name,
        "audio_duration_sec": duration_sec,
        "inference_time_sec": inference_time_sec,
        "rtf": rtf,
        "transcription": transcription,
        "reference": ref_text,
        "wer": wer,
        "cer": cer,
    }

    return record


def print_formatted_result(record: Dict[str, Any]) -> None:
    """
    Print the result to stdout matching the requested format.
    """
    print("-" * 40)
    print(f"Language: {record['language']}")
    print(f"Audio duration: {record['audio_duration_sec']:.2f} sec")
    print(f"Inference time: {record['inference_time_sec']:.2f} sec")
    print(f"RTF: {record['rtf']:.2f}")
    if record["wer"] is not None:
        print(f"WER: {record['wer'] * 100:.2f}% | CER: {record['cer'] * 100:.2f}%")
    print("\nTranscription:")
    print(record["transcription"])
    print("-" * 40)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark local offline IndicConformer STT model."
    )
    parser.add_argument(
        "target",
        type=str,
        help="Path to a WAV audio file or a directory containing WAV files."
    )
    parser.add_argument(
        "--lang", "-l",
        type=str,
        default=None,
        choices=["hindi", "english", "hi", "en"],
        help="Language of the input audio (default: auto-detect from path)."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(DEFAULT_RESULTS_CSV),
        help=f"CSV file path to append results (default: {DEFAULT_RESULTS_CSV})."
    )
    parser.add_argument(
        "--num-threads", "-t",
        type=int,
        default=2,
        help="Number of inference threads for ONNX runtime (default: 2)."
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="int8",
        choices=["int8", "fp32"],
        help="Model precision (default: int8)."
    )

    args = parser.parse_args()
    target_path = Path(args.target)

    if not target_path.exists():
        print(f"Error: Target path '{target_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Build audio file list
    audio_files: List[Path] = []
    if target_path.is_file():
        if target_path.suffix.lower() != ".wav":
            print(f"Error: Target file '{target_path.name}' is not a WAV file.", file=sys.stderr)
            sys.exit(1)
        audio_files.append(target_path)
    elif target_path.is_dir():
        audio_files = sorted(list(target_path.glob("**/*.wav")))
        if not audio_files:
            print(f"Error: No .wav files found in directory '{target_path}'.", file=sys.stderr)
            sys.exit(1)

    config = STTConfig(
        num_threads=args.num_threads,
        results_csv=Path(args.output),
        precision=args.precision
    )

    transcribers: Dict[str, IndicConformerTranscriber] = {}
    records: List[Dict[str, Any]] = []

    print(f"\nRunning Offline STT Benchmark ({len(audio_files)} audio file(s))...")

    for audio_file in audio_files:
        try:
            record = process_single_file(
                audio_path=audio_file,
                transcribers=transcribers,
                user_lang=args.lang,
                config=config
            )
            print_formatted_result(record)
            records.append(record)
        except (AudioProcessingError, TranscriberError, ValueError, FileNotFoundError) as e:
            print(f"\n[ERROR] Processing '{audio_file.name}' failed: {e}", file=sys.stderr)

    if records:
        save_to_csv(Path(args.output), records)
        print(f"\nResults successfully saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
