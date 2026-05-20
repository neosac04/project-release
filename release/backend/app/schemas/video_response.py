from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.schemas.response import AnalysisMetrics, ModelVote


class FrameResult(BaseModel):
    frame_index: int
    timestamp_sec: float
    final_score: float
    face_detected: bool


class VideoDetectionResponse(BaseModel):
    result_id: str
    media_type: Literal["video"] = "video"
    final_score: float
    verdict: Literal["real", "fake"]
    is_uncertain: bool
    frames_analyzed: int
    faces_detected: int
    frame_results: list[FrameResult]
    temporal_consistency: float
    aggregation_strategy: str
    model_votes: dict[str, ModelVote]
    fusion_weights: dict[str, float]
    explanations: list[str]
    total_inference_time_ms: float
    analysis: AnalysisMetrics | None = None
