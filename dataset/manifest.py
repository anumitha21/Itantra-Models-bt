"""
Deterministic Dataset Manifest Generator for AI4Bharat Kathbath Clean & Noisy Test Sets.
Processes extracted audio for Hindi, Tamil, and Telugu, converts m4a to 16kHz mono WAV,
matches reference transcripts, and deterministically generates dataset/manifest.csv.
"""

import os
import sys
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import soundfile as sf

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_backend.core.logging import get_logger

logger = get_logger("ManifestGenerator")

DATASET_ROOT = PROJECT_ROOT / "data" / "kathbath"
MANIFEST_OUTPUT_PATH = PROJECT_ROOT / "dataset" / "manifest.csv"

LANG_MAP = {
    "hindi": "hi",
    "tamil": "ta",
    "telugu": "te",
}


def convert_m4a_to_wav(m4a_path: Path, wav_path: Path, target_sr: int = 16000) -> Optional[float]:
    """
    Convert .m4a audio file to 16kHz mono WAV and return audio duration in seconds.
    """
    if wav_path.exists() and wav_path.stat().st_size > 0:
        try:
            info = sf.info(str(wav_path))
            return info.duration
        except Exception:
            pass

    wav_path.parent.mkdir(parents=True, exist_ok=True)

    if HAS_AV:
        try:
            container = av.open(str(m4a_path))
            audio_stream = next(s for s in container.streams if s.type == "audio")
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
            frames = []
            for packet in container.demux(audio_stream):
                for frame in packet.decode():
                    for resampled in resampler.resample(frame):
                        frames.append(resampled.to_ndarray())
            if not frames:
                return None
            waveform = np.concatenate(frames, axis=1).squeeze(0).astype(np.float32)
            sf.write(str(wav_path), waveform, target_sr)
            return float(len(waveform)) / float(target_sr)
        except Exception as e:
            logger.warning(f"PyAV failed to convert {m4a_path.name}: {e}")

    try:
        data, sr = sf.read(str(m4a_path), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            import scipy.signal
            gcd = np.gcd(int(sr), int(target_sr))
            up = int(target_sr // gcd)
            down = int(sr // gcd)
            data = scipy.signal.resample_poly(data, up, down).astype(np.float32)
        sf.write(str(wav_path), data, target_sr)
        return float(len(data)) / float(target_sr)
    except Exception as e:
        logger.error(f"Failed to convert {m4a_path.name} to WAV: {e}")
        return None


def parse_transcripts(transcript_path: Path) -> Dict[str, str]:
    """
    Parse Kathbath transcription_n2w.txt file mapping audio stem/id -> reference text.
    """
    transcripts = {}
    if not transcript_path.exists():
        return transcripts

    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            if "\t" in line_str:
                parts = line_str.split("\t", 1)
            else:
                parts = line_str.split(" ", 1)

            if len(parts) == 2:
                file_id = Path(parts[0]).stem
                text = parts[1].strip()
                transcripts[file_id] = text
    return transcripts


def build_manifest(
    samples_per_cell: int = 10,
    seed: int = 42,
    output_csv: Path = MANIFEST_OUTPUT_PATH,
    download_remote: bool = False
) -> List[Dict[str, Any]]:
    """
    Deterministic manifest generation across languages (hi, ta, te) × conditions (clean, noisy).
    """
    random.seed(seed)
    manifest_rows = []

    logger.info(f"Generating deterministic manifest (seed={seed}, samples_per_cell={samples_per_cell})...")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    for condition in ["clean", "noisy"]:
        cond_dir = DATASET_ROOT / condition
        audio_dir = cond_dir / "audio"
        transcripts_dir = cond_dir / "transcripts"
        wav_cache_dir = cond_dir / "wav_16k"

        # Load all transcripts for this condition
        cond_transcripts: Dict[str, Dict[str, str]] = {"hi": {}, "ta": {}, "te": {}}
        for full_lang, lang_code in LANG_MAP.items():
            for t_file in transcripts_dir.rglob("*.txt"):
                if full_lang in t_file.parts or lang_code in t_file.parts or full_lang in t_file.name:
                    cond_transcripts[lang_code].update(parse_transcripts(t_file))
            logger.info(f"Loaded {len(cond_transcripts[lang_code])} reference transcripts for [{lang_code}] [{condition}].")

        for full_lang, lang_code in LANG_MAP.items():
            # Find all available audio files for this language
            lang_audio_files = []
            for ext in ["*.m4a", "*.wav"]:
                for a_file in audio_dir.rglob(ext):
                    if full_lang in a_file.parts or lang_code in a_file.parts:
                        lang_audio_files.append(a_file)

            if not lang_audio_files:
                logger.warning(f"No Kathbath audio found for {lang_code} [{condition}]. Checking fallback test_audio...")
                local_dir = PROJECT_ROOT / "test_audio" / ("hindi" if lang_code == "hi" else full_lang)
                if local_dir.exists():
                    for a_file in local_dir.glob("*.wav"):
                        ref_file = a_file.with_suffix(".txt")
                        ref_text = ref_file.read_text(encoding="utf-8").strip() if ref_file.exists() else ""
                        info = sf.info(str(a_file))
                        manifest_rows.append({
                            "audio_path": str(a_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                            "duration": round(info.duration, 3),
                            "language": lang_code,
                            "dataset_name": "local_sample",
                            "split": "test",
                            "noise_condition": condition,
                            "speaker_id": "spk01",
                            "reference_text": ref_text,
                        })
                continue

            # Deterministically sort and sample
            lang_audio_files.sort(key=lambda p: p.name)
            # Filter files that have reference text if possible
            matched_files = [f for f in lang_audio_files if f.stem in cond_transcripts[lang_code]]
            candidate_pool = matched_files if len(matched_files) >= samples_per_cell else lang_audio_files
            sampled_files = random.sample(candidate_pool, min(samples_per_cell, len(candidate_pool)))

            logger.info(f"Converting and processing {len(sampled_files)} samples for [{lang_code}] [{condition}]...")

            for audio_file in sampled_files:
                stem = audio_file.stem
                split_name = "test_known" if "testkn" in str(audio_file) or "test_known" in str(audio_file) else "test"
                speaker_id = stem.split("-")[1] if "-" in stem else (stem.split("_")[0] if "_" in stem else "spk")
                ref_text = cond_transcripts[lang_code].get(stem, "")

                if audio_file.suffix.lower() == ".m4a":
                    wav_file = wav_cache_dir / full_lang / f"{stem}.wav"
                    duration = convert_m4a_to_wav(audio_file, wav_file)
                    final_path = wav_file
                else:
                    final_path = audio_file
                    info = sf.info(str(audio_file))
                    duration = info.duration

                if duration is None:
                    continue

                manifest_rows.append({
                    "audio_path": str(final_path.relative_to(PROJECT_ROOT) if final_path.is_relative_to(PROJECT_ROOT) else final_path).replace("\\", "/"),
                    "duration": round(duration, 3),
                    "language": lang_code,
                    "dataset_name": "kathbath",
                    "split": split_name,
                    "noise_condition": condition,
                    "speaker_id": speaker_id,
                    "reference_text": ref_text,
                })

    # Save manifest CSV
    fieldnames = [
        "audio_path",
        "duration",
        "language",
        "dataset_name",
        "split",
        "noise_condition",
        "speaker_id",
        "reference_text",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    logger.info(f"Manifest written successfully: {output_csv} ({len(manifest_rows)} total samples).")
    return manifest_rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate benchmark manifest.")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples per language/condition cell.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument("--output", type=str, default=str(MANIFEST_OUTPUT_PATH), help="Output manifest CSV path.")
    args = parser.parse_args()

    build_manifest(
        samples_per_cell=args.samples,
        seed=args.seed,
        output_csv=Path(args.output),
    )
