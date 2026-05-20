"""
Temporal score aggregation for video deepfake detection.
Adapted from dfdc_deepfake_challenge confident_strategy.
"""
from __future__ import annotations

from typing import List


def confident_strategy(
    scores: List[float],
    high_conf_threshold: float = 0.8,
    min_high_conf: int = 2,
) -> float:
    """
    If >= min_high_conf frames exceed high_conf_threshold, return their mean.
    Otherwise return the overall mean across all frames.
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
