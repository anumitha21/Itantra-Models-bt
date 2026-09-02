"""
Unit Tests for Manifest Generator (dataset/manifest.py).
"""

import pytest
import csv
from pathlib import Path
from dataset.manifest import build_manifest, LANG_MAP


def test_manifest_lang_mapping():
    assert LANG_MAP["hindi"] == "hi"
    assert LANG_MAP["tamil"] == "ta"
    assert LANG_MAP["telugu"] == "te"


def test_build_manifest_local_fallback(tmp_path):
    output_csv = tmp_path / "test_manifest.csv"
    manifest_rows = build_manifest(
        samples_per_cell=2,
        seed=42,
        output_csv=output_csv,
        download_remote=False
    )
    assert output_csv.exists()
    assert len(manifest_rows) > 0

    with open(output_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        assert "audio_path" in fields
        assert "language" in fields
        assert "noise_condition" in fields
        assert "reference_text" in fields
