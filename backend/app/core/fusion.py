"""
Adaptive confidence-weighted fusion for the 5-model ensemble.

Models:
  vit         — ViT-Base HF dima806, AUC 0.999   (global patterns)
  siglip      — SigLIP-Base fine-tuned, 94.44%   (custom-trained binary)
  f3net       — Frequency DCT, AUC 0.958          (GAN/diffusion artifacts)
  efficientnet— Face-texture CNN, AUC 0.764       (micro-texture forensics)
  hive        — External Hive AI API oracle        (conditional on api key)
"""
from __future__ import annotations

from dataclasses import dataclass

_BASE_WEIGHTS_FACE = {
    "vit":          0.35,
    "siglip":       0.25,
    "f3net":        0.20,
    "efficientnet": 0.15,
    "hive":         0.05,
    "xceptionnet":  0.00,
}

_BASE_WEIGHTS_NO_FACE = {
    "vit":          0.35,
    "siglip":       0.25,
    "f3net":        0.25,
    "efficientnet": 0.10,
    "hive":         0.05,
    "xceptionnet":  0.00,
}

UNCERTAINTY_LOW  = 0.38
UNCERTAINTY_HIGH = 0.62
_CONFIDENCE_BONUS = 0.10


@dataclass
class FusionResult:
    final_score: float
    weights: dict[str, float]
    is_uncertain: bool


def fuse(model_scores: dict[str, float], face_detected: bool) -> FusionResult:
    base = _BASE_WEIGHTS_FACE if face_detected else _BASE_WEIGHTS_NO_FACE

    adjusted: dict[str, float] = {}
    for name, score in model_scores.items():
        w = base.get(name, 0.0)
        if w == 0.0:
            continue
        adjusted[name] = w + abs(score - 0.5) * _CONFIDENCE_BONUS

    total = sum(adjusted.values())
    if total <= 0 or not adjusted:
        return FusionResult(final_score=0.5, weights={}, is_uncertain=True)

    weights = {name: w / total for name, w in adjusted.items()}
    final_score = sum(weights[name] * model_scores[name] for name in weights)
    is_uncertain = UNCERTAINTY_LOW <= final_score <= UNCERTAINTY_HIGH

    return FusionResult(final_score=float(final_score), weights=weights, is_uncertain=is_uncertain)
