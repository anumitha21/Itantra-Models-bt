"""
Unit Tests for Benchmark Metrics (src/ai_backend/benchmark/metrics.py).
"""

from ai_backend.benchmark.metrics import compute_accuracy_metrics


def test_compute_metrics_exact_match():
    ref = "तीन लोग उत्तर दिशा में हैं"
    hyp = "तीन लोग उत्तर दिशा में हैं"
    wer, cer = compute_accuracy_metrics(ref, hyp)

    assert wer == 0.0
    assert cer == 0.0


def test_compute_metrics_no_reference():
    wer, cer = compute_accuracy_metrics(None, "some hypothesis")
    assert wer is None
    assert cer is None

    wer_empty, cer_empty = compute_accuracy_metrics("", "some hypothesis")
    assert wer_empty is None
    assert cer_empty is None
