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


def test_composite_score_calculation():
    from ai_backend.benchmark.metrics import calculate_weighted_overall_score, compute_composite_score
    score = calculate_weighted_overall_score(
        wer=0.10,
        latency_warm_sec=0.35,
        ram_mb=800.0,
        model_size_mb=180.0,
    )
    assert 0.0 <= score <= 100.0
    assert compute_composite_score(0.10, 0.35, 800.0, 180.0) == score


