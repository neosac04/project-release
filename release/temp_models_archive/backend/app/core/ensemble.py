from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from app.models.base import ModelOutput


@dataclass
class EnsembleOutput:
    verdict: str
    fake_probability: float
    confidence: float
    disagreement: float
    weights_used: dict[str, float]


class WeightedEnsemble:
    """
    Type-adaptive weighted ensemble.
    Default weights are calibrated on diverse fake datasets.
    When the FakeTypeClassifier provides a confident type hint,
    weights shift to favour the specialist model for that forgery type.
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "univfd":       0.45,
        "efficientnet": 0.30,
        "xception":     0.25,
    }

    # Specialist weight profiles per forgery type
    TYPE_WEIGHTS: dict[str, dict[str, float]] = {
        "diffusion":       {"univfd": 0.50, "xception": 0.30, "efficientnet": 0.20},
        "gan":             {"xception": 0.40, "univfd": 0.35, "efficientnet": 0.25},
        "face_swap":       {"efficientnet": 0.40, "xception": 0.35, "univfd": 0.25},
        "face_reenactment":{"efficientnet": 0.40, "xception": 0.35, "univfd": 0.25},
        "photoshop":       {"xception": 0.40, "efficientnet": 0.35, "univfd": 0.25},
    }

    def combine(
        self,
        predictions: list[ModelOutput],
        type_hint: str | None = None,
        fake_threshold: float = 0.60,
        real_threshold: float = 0.40,
    ) -> EnsembleOutput:
        if not predictions:
            return EnsembleOutput(
                verdict="UNCERTAIN",
                fake_probability=0.5,
                confidence=0.0,
                disagreement=0.0,
                weights_used=self.DEFAULT_WEIGHTS,
            )

        weights = self.TYPE_WEIGHTS.get(type_hint or "", self.DEFAULT_WEIGHTS)

        # Normalise weights to models that actually ran
        available = {p.model_name for p in predictions}
        active_w = {k: v for k, v in weights.items() if k in available}
        total_w = sum(active_w.values()) or 1.0
        active_w = {k: v / total_w for k, v in active_w.items()}

        weighted_fake = sum(
            active_w.get(p.model_name, 0.0) * p.fake_prob
            for p in predictions
        )

        probs = [p.fake_prob for p in predictions]
        disagreement = float(np.std(probs)) if len(probs) > 1 else 0.0

        if weighted_fake >= fake_threshold:
            verdict = "FAKE"
        elif weighted_fake <= real_threshold:
            verdict = "REAL"
        else:
            verdict = "UNCERTAIN"

        # Downgrade verdict if models strongly disagree
        if disagreement > 0.30 and verdict != "UNCERTAIN":
            verdict = "UNCERTAIN"

        confidence = abs(weighted_fake - 0.5) * 2.0

        return EnsembleOutput(
            verdict=verdict,
            fake_probability=float(weighted_fake),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            disagreement=disagreement,
            weights_used=active_w,
        )
