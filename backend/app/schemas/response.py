from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ModelVote(BaseModel):
    fake_prob: float
    real_prob: float
    inference_time_ms: float


class DetectionResponse(BaseModel):
    result_id: str
    final_score: float
    verdict: Literal["real", "fake"]
    face_detected: bool
    is_uncertain: bool

    # Per-model fake probabilities + timings (keys: "efficientnet", "xceptionnet", "f3net",
    # later "spectral_analyzer" for the masked Hive fallback)
    model_votes: dict[str, ModelVote]

    # Normalised fusion weights actually applied this prediction
    fusion_weights: dict[str, float]

    explanations: list[str]
    total_inference_time_ms: float
