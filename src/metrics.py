"""
Accuracy Metrics for STT Baseline (WER & CER Calculation).
"""

from typing import Optional, Tuple
import jiwer


def compute_metrics(
    reference: Optional[str],
    hypothesis: str
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute Word Error Rate (WER) and Character Error Rate (CER) given reference and hypothesis texts.

    Args:
        reference: Reference ground-truth text string (or None if unavailable).
        hypothesis: Model transcribed text string.

    Returns:
        Tuple of (wer, cer) as floats, or (None, None) if reference is None/empty.
    """
    if not reference or not reference.strip():
        return None, None

    ref_str = reference.strip()
    hyp_str = hypothesis.strip()

    try:
        wer_val = float(jiwer.wer(ref_str, hyp_str))
        cer_val = float(jiwer.cer(ref_str, hyp_str))
        return wer_val, cer_val
    except Exception:
        # Fallback if jiwer fails on unexpected input
        return None, None
