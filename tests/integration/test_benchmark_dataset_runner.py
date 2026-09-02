"""
Integration Tests for BenchmarkDatasetRunner (benchmark/dataset_runner.py).
"""

import pytest
import pandas as pd
from pathlib import Path
from benchmark.dataset_runner import BenchmarkDatasetRunner
from dataset.manifest import build_manifest


def test_benchmark_runner_mock_manifest(tmp_path):
    manifest_csv = tmp_path / "mock_manifest.csv"
    results_csv = tmp_path / "results.csv"

    # Build local manifest
    build_manifest(samples_per_cell=1, seed=42, output_csv=manifest_csv, download_remote=False)
    assert manifest_csv.exists()

    runner = BenchmarkDatasetRunner(
        manifest_path=manifest_csv,
        results_csv_path=results_csv,
    )

    # Run for IndicConformer on Hindi clean
    summaries = runner.run_benchmark(
        models=["indicconformer"],
        languages=["hi"],
        conditions=["clean"],
        run_id="test_run_01"
    )

    assert len(summaries) >= 1
    assert results_csv.exists()

    df = pd.read_csv(results_csv)
    assert "model_name" in df.columns
    assert "runtime" in df.columns
    assert "precision" in df.columns
    assert "latency_cold_sec" in df.columns
    assert "latency_warm_sec" in df.columns
    assert "rtf" in df.columns
    assert "ram_mb" in df.columns
    assert "score" in df.columns
    assert (df["model_name"] == "indicconformer").any()
