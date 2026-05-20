"""Video deepfake detection endpoint."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config.settings import settings
from app.schemas.video_response import VideoDetectionResponse
from app.storage.result_cache import ResultCache
from app.video.pipeline import VideoPipeline

router = APIRouter()
_video_pipeline: VideoPipeline | None = None

_VIDEO_CONTENT_TYPES = {
    "video/mp4", "video/webm", "video/avi", "video/quicktime",
    "video/x-msvideo", "video/mpeg", "video/x-matroska",
    "video/ogg", "video/3gpp", "application/octet-stream",
}


def _get_pipeline() -> VideoPipeline:
    global _video_pipeline
    if _video_pipeline is None:
        _video_pipeline = VideoPipeline()
    return _video_pipeline


@router.post("/detect/video", response_model=VideoDetectionResponse)
async def detect_video(file: UploadFile = File(...)) -> VideoDetectionResponse:
    content_type = (file.content_type or "").lower()
    if not content_type.startswith("video/") and content_type not in _VIDEO_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"File must be a video. Received content-type: '{content_type}'.",
        )

    video_bytes = await file.read()
    max_bytes = settings.max_video_size_mb * 1024 * 1024

    if len(video_bytes) > max_bytes:
        size_mb = len(video_bytes) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Video too large ({size_mb:.1f} MB). Maximum is {settings.max_video_size_mb} MB.",
        )

    if len(video_bytes) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        pipeline = _get_pipeline()
        result = await pipeline.run(video_bytes, num_frames=settings.video_frames_to_sample)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    ResultCache.get_instance().set(result.result_id, result)
    return result


@router.get("/video/result/{result_id}", response_model=VideoDetectionResponse)
async def get_video_result(result_id: str) -> VideoDetectionResponse:
    result = ResultCache.get_instance().get(result_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found or expired.")
    if not hasattr(result, "media_type") or result.media_type != "video":
        raise HTTPException(status_code=404, detail="Result is not a video result.")
    return result  # type: ignore[return-value]
