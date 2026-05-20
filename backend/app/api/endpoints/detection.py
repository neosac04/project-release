from __future__ import annotations
import io
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from app.config.settings import settings
from app.core.pipeline import DetectionPipeline
from app.schemas.response import DetectionResponse
from app.storage.result_cache import ResultCache

router = APIRouter()
_pipeline: DetectionPipeline | None = None


def get_pipeline() -> DetectionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DetectionPipeline()
    return _pipeline


@router.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    # Validate content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="File must be an image.")

    data = await file.read()
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size_mb} MB.",
        )

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Cannot decode image file.")

    pipeline = get_pipeline()
    result = await pipeline.run(image)

    ResultCache.get_instance().set(result.result_id, result, image_bytes=data)
    return result


@router.get("/result/{result_id}", response_model=DetectionResponse)
async def get_result(result_id: str):
    result = ResultCache.get_instance().get(result_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found or expired.")
    return result
