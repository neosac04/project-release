from __future__ import annotations

import io

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

from app.explainability.overlay import ensemble_heatmap, heatmap_to_overlay
from app.models.registry import ModelRegistry
from app.preprocessing.face_detection import detect_largest_face
from app.preprocessing.image_transforms import preprocess
from app.storage.result_cache import ResultCache


router = APIRouter()

# Which models run on the face crop vs the full image (mirrors pipeline.py)
_FULL_IMAGE_MODELS = {"f3net", "vit"}
_VALID_MODELS = {"efficientnet", "xceptionnet", "f3net", "vit", "ensemble"}


@router.get("/heatmap/{result_id}/{model_name}")
async def get_heatmap(result_id: str, model_name: str):
    result = ResultCache.get_instance().get(result_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found or expired.")

    if model_name not in _VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"model_name must be one of {_VALID_MODELS}")

    cache = ResultCache.get_instance()
    image_bytes = getattr(cache, "_images", {}).get(result_id)
    if image_bytes is None:
        raise HTTPException(
            status_code=404,
            detail="Heatmap image data not available. Re-submit the image.",
        )

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    face_crop = detect_largest_face(np.array(image))
    face_image = Image.fromarray(face_crop) if face_crop is not None else image

    face_input = preprocess(face_image)
    full_input = preprocess(image)

    registry = ModelRegistry.get_instance()

    if model_name == "ensemble":
        # Same weighting as the fusion engine, simplified to a fixed mix
        # Mirrors fusion.py base weights (ViT dominates, EfficientNet supports).
        ensemble_weights = (
            {"vit": 0.70, "efficientnet": 0.30}
            if result.face_detected
            else {"vit": 0.80, "efficientnet": 0.20}
        )
        heatmaps: list[np.ndarray] = []
        weights: list[float] = []
        for name, weight in ensemble_weights.items():
            det = registry.get(name)
            if det is None:
                continue
            chosen_input = full_input if name in _FULL_IMAGE_MODELS else face_input
            hm = det.get_heatmap(chosen_input)
            if hm is not None:
                heatmaps.append(hm)
                weights.append(weight)
        if not heatmaps:
            raise HTTPException(status_code=503, detail="No model produced a heatmap.")
        heatmap = ensemble_heatmap(heatmaps, weights)
    else:
        det = registry.get(model_name)
        if det is None:
            raise HTTPException(status_code=503, detail=f"Model '{model_name}' not loaded.")
        chosen_input = full_input if model_name in _FULL_IMAGE_MODELS else face_input
        heatmap = det.get_heatmap(chosen_input)
        if heatmap is None:
            raise HTTPException(status_code=503, detail=f"Model '{model_name}' does not support heatmaps.")

    png_bytes = heatmap_to_overlay(image, heatmap)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")
