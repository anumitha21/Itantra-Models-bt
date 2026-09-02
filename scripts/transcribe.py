"""
CLI Script to transcribe a single WAV file using AI Backend.
Usage:
    python scripts/transcribe.py --language hi --audio test_audio/hindi/test01.wav --model indicconformer
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

from ai_backend.core.types import AudioInput
from ai_backend.pipeline.speech_pipeline import SpeechPipeline


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio file using AI Backend.")
    parser.add_argument("--audio", "-a", type=str, required=True, help="Path to input WAV file.")
    parser.add_argument("--language", "-l", type=str, default="hi", help="Language code (default: hi).")
    parser.add_argument("--model", "-m", type=str, default=None, help="STT Model name (e.g. indicconformer, whisper_tiny, whisper_small, mms).")

    args = parser.parse_args()
    audio_path = Path(args.audio)

    if not audio_path.exists():
        print(f"Error: Audio file '{audio_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        audio = AudioInput.from_wav_file(audio_path)
        pipeline = SpeechPipeline()
        result = pipeline.transcribe(audio, language=args.language, model_name=args.model)

        print("-" * 40)
        print(f"Language: {result.language}")
        print(f"Model: {result.model_name}")
        print(f"Quantization: {result.quantization}")
        print(f"Audio duration: {result.audio_duration_sec:.2f} sec")
        print(f"Inference time: {result.inference_time_sec:.2f} sec")
        print(f"RTF: {result.rtf:.2f}")
        print("\nTranscription:")
        print(result.text)
        print("-" * 40)

    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
