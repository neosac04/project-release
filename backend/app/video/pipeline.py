"""
End-to-end video deepfake detection pipeline.

Flow:
  1. Extract num_frames evenly-spaced frames (OpenCV).
  2. For each frame: detect largest face; use face crop if found.
  3. Run the existing DetectionPipeline on each frame image.
  4. Aggregate per-frame scores via confident_strategy (from dfdc).
  5. Return VideoDetectionResponse with per-frame detail and video-level verdict.
"""
from __future__ import annotations

import time
import uuid
from typing import List

import numpy as np
from PIL import Image

from app.analysis.suite import run_analysis
from app.core.pipeline import DetectionPipeline, _display_score, _to_analysis_metrics
from app.preprocessing.face_detection import detect_largest_face
from app.schemas.response import ModelVote
from app.schemas.video_response import FrameResult, VideoDetectionResponse
from app.video.frame_extractor import extract_frames
from app.video.temporal_aggregator import aggregate

# For video analysis we skip ViT and Hive per frame:
#   • ViT is trained on AI-generated imagery (GAN/diffusion), not face-swaps — it gives ~0%
#     for FaceForensics++ content and drags the fusion score into the real zone.
#   • Hive VLM has a 30-second HTTP timeout; 32 frames × 30 s = 16 minutes.
# Only EfficientNet, F3Net, and SigLIP run per frame.  Their weights are re-normalised
# inside fuse() automatically when ViT/Hive are absent.
_VIDEO_SKIP_MODELS: frozenset[str] = frozenset({"vit", "hive"})

# Raw fusion threshold above which a video is classified as fake.
# Lower than the image threshold (0.62) because, without ViT, the
# three remaining models produce moderate-confidence scores (~0.50–0.65)
# for face-swap deepfakes.
_VIDEO_FAKE_THRESHOLD: float = 0.50


class VideoPipeline:
    def __init__(self) -> None:
        self._image_pipeline = DetectionPipeline()

    async def run(
        self,
        video_bytes: bytes,
        num_frames: int = 32,
    ) -> VideoDetectionResponse:
        t_start = time.time()

        frames, fps, frame_indices = extract_frames(video_bytes, num_frames=num_frames)
        if not frames:
            raise ValueError("Could not extract any frames from the video.")

        frame_results: List[FrameResult] = []
        per_frame_scores: List[float] = []
        per_frame_votes: List[dict] = []
        faces_detected_count = 0

        for frame, frame_idx in zip(frames, frame_indices):
            timestamp_sec = frame_idx / fps

            frame_np = np.array(frame.convert("RGB"))
            face_crop = detect_largest_face(frame_np)
            face_detected = face_crop is not None

            image_to_analyse: Image.Image = (
                Image.fromarray(face_crop) if face_detected else frame
            )

            # Run with raw scores (apply_display=False) so aggregation works on
            # un-mapped probabilities.  Skip ViT (wrong domain) and Hive (too slow).
            frame_det = await self._image_pipeline.run(
                image_to_analyse,
                apply_display=False,
                skip_models=_VIDEO_SKIP_MODELS,
            )
            raw_score = frame_det.final_score          # raw fusion score
            per_frame_scores.append(raw_score)
            per_frame_votes.append({
                name: vote.fake_prob for name, vote in frame_det.model_votes.items()
            })

            if face_detected:
                faces_detected_count += 1

            # Per-frame timeline shows display-mapped score for UI consistency
            frame_display = _display_score(raw_score, _VIDEO_FAKE_THRESHOLD)
            frame_results.append(FrameResult(
                frame_index=frame_idx,
                timestamp_sec=round(timestamp_sec, 3),
                final_score=round(frame_display, 4),
                face_detected=face_detected,
            ))

        # Aggregate raw per-frame scores, then apply display mapping once.
        final_score_raw = aggregate(per_frame_scores, strategy="confident")
        final_score = _display_score(final_score_raw, _VIDEO_FAKE_THRESHOLD)

        temporal_consistency = (
            float(np.std(per_frame_scores)) if len(per_frame_scores) > 1 else 0.0
        )

        all_models: set[str] = set()
        for v in per_frame_votes:
            all_models.update(v.keys())

        aggregated_votes: dict[str, ModelVote] = {}
        for model_name in sorted(all_models):
            model_scores = [v.get(model_name, 0.5) for v in per_frame_votes]
            mean_fake_raw = float(np.mean(model_scores))
            mean_fake_display = _display_score(mean_fake_raw, _VIDEO_FAKE_THRESHOLD)
            aggregated_votes[model_name] = ModelVote(
                fake_prob=round(mean_fake_display, 4),
                real_prob=round(1.0 - mean_fake_display, 4),
                inference_time_ms=0.0,
            )

        # Verdict and uncertainty use the raw aggregated score.
        verdict: str = "fake" if final_score_raw >= _VIDEO_FAKE_THRESHOLD else "real"
        is_uncertain = 0.38 <= final_score_raw < _VIDEO_FAKE_THRESHOLD

        # Analysis on middle frame (best-effort)
        analysis_metrics = None
        try:
            mid_frame = frames[len(frames) // 2]
            analysis_result = run_analysis(mid_frame.convert("RGB"), saliency=None)
            analysis_metrics = _to_analysis_metrics(analysis_result)
        except Exception as exc:
            print(f"⚠️ Video analysis suite failed: {exc!r}")

        total_ms = (time.time() - t_start) * 1000

        explanations = _build_explanations(
            final_score=final_score,
            final_score_raw=final_score_raw,
            frames_analyzed=len(frames),
            faces_detected=faces_detected_count,
            temporal_consistency=temporal_consistency,
            per_frame_scores=per_frame_scores,
            is_uncertain=is_uncertain,
        )

        return VideoDetectionResponse(
            result_id=str(uuid.uuid4()),
            final_score=round(final_score, 4),
            verdict=verdict,  # type: ignore[arg-type]
            is_uncertain=is_uncertain,
            frames_analyzed=len(frames),
            faces_detected=faces_detected_count,
            frame_results=frame_results,
            temporal_consistency=round(temporal_consistency, 4),
            aggregation_strategy="confident",
            model_votes=aggregated_votes,
            fusion_weights={},
            explanations=explanations,
            total_inference_time_ms=round(total_ms, 1),
            analysis=analysis_metrics,
        )


def _build_explanations(
    final_score: float,
    final_score_raw: float,
    frames_analyzed: int,
    faces_detected: int,
    temporal_consistency: float,
    per_frame_scores: List[float],
    is_uncertain: bool,
) -> List[str]:
    """Build human-readable explanations for a video detection result.

    Args:
        final_score: Display-mapped final score (shown in UI).
        final_score_raw: Raw aggregated score before display mapping (used for logic).
        per_frame_scores: Raw per-frame scores (for frame-level statistics).
    """
    explanations: List[str] = []
    pct = round(final_score * 100)

    if final_score_raw >= _VIDEO_FAKE_THRESHOLD:
        explanations.append(
            f"Video analysis across {frames_analyzed} frames: {pct}% likely a deepfake. "
            + ("Strong and consistent manipulation signals detected."
               if pct >= 88 else "Moderate manipulation signals — treat with caution.")
        )
    else:
        real_pct = 100 - pct
        explanations.append(
            f"Video analysis across {frames_analyzed} frames: {real_pct}% likely authentic. "
            + ("No significant manipulation signals detected."
               if real_pct >= 70 else "Minimal manipulation signals — borderline result.")
        )

    face_pct = round(faces_detected / frames_analyzed * 100) if frames_analyzed else 0
    explanations.append(
        f"Faces detected in {faces_detected}/{frames_analyzed} frames ({face_pct}%). "
        + ("Face crops were used for analysis where available."
           if faces_detected > 0 else "Full frames were analysed — no faces found.")
    )

    if temporal_consistency < 0.08:
        explanations.append(
            f"High temporal consistency (std={temporal_consistency:.3f}) — "
            "the deepfake signal is uniform across the video."
        )
    elif temporal_consistency > 0.20:
        explanations.append(
            f"High temporal variation (std={temporal_consistency:.3f}) — "
            "scores vary significantly between frames, suggesting partial manipulation."
        )

    # Count raw per-frame scores above the fake threshold
    high_fake_count = sum(1 for s in per_frame_scores if s > _VIDEO_FAKE_THRESHOLD)
    if high_fake_count > 0:
        explanations.append(
            f"{high_fake_count}/{frames_analyzed} frame(s) showed manipulation signals "
            f"above the fake threshold ({round(_VIDEO_FAKE_THRESHOLD * 100)}%)."
        )

    if is_uncertain:
        explanations.append(
            "⚠ Signals are mixed — borderline result. Human review is recommended."
        )

    return explanations
