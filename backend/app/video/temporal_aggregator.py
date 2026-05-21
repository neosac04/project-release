"""
Temporal score aggregation for video deepfake detection.
Adapted from dfdc_deepfake_challenge confident_strategy.
"""
from __future__ import annotations

from typing import List


def confident_strategy(
    scores: List[float],
    high_conf_threshold: float = 0.55,
    min_high_conf: int = 2,
) -> float:
    """
    If >= min_high_conf frames exceed high_conf_threshold, return their mean.
    Otherwise return the overall mean across all frames.

    Threshold lowered from 0.80 to 0.55: the video pipeline now operates on raw
    scores (no temperature scaling) from only EfficientNet + F3Net + SigLIP.
    FaceForensics++ face-swap deepfakes typically score 0.55–0.70 with these
    three models, so 0.55 is a more appropriate confidence boundary than 0.80.
    """
    if not scores:
        return 0.5
    high_conf = [s for s in scores if s >= high_conf_threshold]
    if len(high_conf) >= min_high_conf:
        return float(sum(high_conf) / len(high_conf))
    return float(sum(scores) / len(scores))


def aggregate(scores: List[float], strategy: str = "confident") -> float:
    if strategy == "confident":
        return confident_strategy(scores)
    return float(sum(scores) / len(scores)) if scores else 0.5
