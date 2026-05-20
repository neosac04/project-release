from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ModelVote(BaseModel):
    fake_prob: float
    real_prob: float
    inference_time_ms: float


class FFTMetrics(BaseModel):
    low_freq: float
    mid_freq: float
    high_freq: float
    spectral_irregularity: float
    profile: list[float] = []


class TextureMetrics(BaseModel):
    sharpness: float
    texture_uniformity: float
    noise_level: float
    compression_artifacts: float


class SkinMetrics(BaseModel):
    pore_detail: float
    blotchiness: float
    edge_blend: float


class AnalysisMetrics(BaseModel):
    fft: FFTMetrics
    texture: TextureMetrics
    symmetry_score: float | None = None
    skin: SkinMetrics | None = None
    top_attention_regions: list[str] = []
    region_scores: dict[str, dict[str, float]] = {}


class DetectionResponse(BaseModel):
    result_id: str
    media_type: Literal["image"] = "image"
    final_score: float
    verdict: Literal["real", "fake"]
    face_detected: bool
    is_uncertain: bool
    model_votes: dict[str, ModelVote]
    fusion_weights: dict[str, float]
    explanations: list[str]
    total_inference_time_ms: float
    analysis: AnalysisMetrics | None = None
