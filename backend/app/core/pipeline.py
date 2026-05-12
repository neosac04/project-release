from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.fusion import fuse
from app.models.registry import ModelRegistry
from app.preprocessing.face_detection import detect_largest_face
from app.preprocessing.image_transforms import preprocess
from app.schemas.response import DetectionResponse, ModelVote


# Which models get the face crop vs. the full image.
# ViT was trained on uncropped images → use full image.
_FACE_CROP_MODELS = {"efficientnet", "xceptionnet"}
_FULL_IMAGE_MODELS = {"f3net", "vit"}


class DetectionPipeline:
    def __init__(self) -> None:
        self._registry = ModelRegistry.get_instance()

    async def run(self, image: Image.Image) -> DetectionResponse:
        t_start = time.time()

        source_rgb = np.array(image.convert("RGB"))
        face_crop = detect_largest_face(source_rgb)
        face_detected = face_crop is not None

        full_image = image.convert("RGB")
        face_image = Image.fromarray(face_crop) if face_detected else full_image

        # Two preprocess dicts — share work where models agree on input view
        face_input = preprocess(face_image)
        full_input = preprocess(full_image)

        # Dispatch all loaded detectors in parallel via a thread pool.
        loop = asyncio.get_event_loop()
        tasks: dict[str, asyncio.Future] = {}
        for name, det in self._registry.all().items():
            chosen_input = full_input if name in _FULL_IMAGE_MODELS else face_input
            tasks[name] = loop.run_in_executor(None, det.predict, chosen_input)

        outputs = {name: await fut for name, fut in tasks.items()}

        if not outputs:
            raise RuntimeError("No detection models are loaded.")

        # Build model votes
        model_votes: dict[str, ModelVote] = {
            name: ModelVote(
                fake_prob=out.fake_prob,
                real_prob=out.real_prob,
                inference_time_ms=out.inference_time_ms,
            )
            for name, out in outputs.items()
        }

        # Fusion
        scores = {name: out.fake_prob for name, out in outputs.items()}
        fusion = fuse(scores, face_detected=face_detected)

        verdict: str = "fake" if fusion.final_score >= 0.5 else "real"

        explanations = self._build_explanations(
            scores=scores,
            face_detected=face_detected,
            is_uncertain=fusion.is_uncertain,
            final_score=fusion.final_score,
        )

        total_ms = (time.time() - t_start) * 1000

        return DetectionResponse(
            result_id=str(uuid.uuid4()),
            final_score=float(fusion.final_score),
            verdict=verdict,  # type: ignore[arg-type]
            face_detected=face_detected,
            is_uncertain=fusion.is_uncertain,
            model_votes=model_votes,
            fusion_weights=fusion.weights,
            explanations=explanations,
            total_inference_time_ms=total_ms,
        )

    def detect_image(self, image_path: str) -> dict:
        image = Image.open(Path(image_path)).convert("RGB")
        result = asyncio.run(self.run(image))
        return result.model_dump()

    def _build_explanations(
        self,
        scores: dict[str, float],
        face_detected: bool,
        is_uncertain: bool,
        final_score: float,
    ) -> list[str]:
        explanations: list[str] = []

        if not face_detected:
            explanations.append("No face detected; face-based models ran on the full image as fallback")

        eff = scores.get("efficientnet")
        xcep = scores.get("xceptionnet")
        f3 = scores.get("f3net")
        vit = scores.get("vit")

        if vit is not None and vit >= 0.6:
            explanations.append("ViT flagged learned synthesis patterns across the full image")
        if eff is not None and eff >= 0.6:
            explanations.append("EfficientNet flagged facial texture inconsistencies")
        if xcep is not None and xcep >= 0.6:
            explanations.append("XceptionNet detected localised manipulation artifacts")
        if f3 is not None and f3 >= 0.6:
            explanations.append("F3Net detected frequency-domain artifacts characteristic of synthesis")

        if is_uncertain:
            explanations.append(
                f"Final score {final_score:.2f} falls in the uncertainty band — verdict is low confidence"
            )

        if not explanations:
            explanations.append("No strong manipulation indicators detected across the model trio")

        return explanations
