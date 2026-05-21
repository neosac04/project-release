from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image


# ── Display score mapping ──────────────────────────────────────────────────────
# Maps raw fusion scores into a human-believable display range:
#   Fake region  (raw > fake_threshold) → [0.86, 0.96]
#   Real region  (raw < 0.38)           → [0.04, 0.14]
#   Uncertain    (0.38 – fake_threshold) → [0.40, 0.60]
#
# fake_threshold defaults to 0.62 for images.
# Pass fake_threshold=0.50 for video aggregation (see video/pipeline.py).


def _display_score(raw: float, fake_threshold: float = 0.62) -> float:
    """Map a raw probability into the capped display range."""
    if raw > fake_threshold:
        t = min((raw - fake_threshold) / max(1.0 - fake_threshold, 1e-9), 1.0)
        return 0.86 + t * 0.10          # → [0.86, 0.96]
    elif raw < 0.38:
        t = raw / 0.38
        return 0.04 + t * 0.10          # → [0.04, 0.14]
    else:
        uncertain_width = max(fake_threshold - 0.38, 1e-9)
        t = (raw - 0.38) / uncertain_width
        return 0.40 + t * 0.20          # → [0.40, 0.60]

from app.analysis.suite import AnalysisResult, run_analysis
from app.core.fusion import fuse
from app.models.registry import ModelRegistry
from app.preprocessing.face_detection import detect_largest_face
from app.preprocessing.image_transforms import preprocess
from app.schemas.response import (
    AnalysisMetrics,
    DetectionResponse,
    FFTMetrics,
    ModelVote,
    SkinMetrics,
    TextureMetrics,
)


# Which models get the face crop vs. the full image.
# ViT / F3Net / Hive operate on the full image.
# EfficientNet / XceptionNet / SigLIP use the face crop.
_FACE_CROP_MODELS = {"efficientnet", "xceptionnet", "siglip"}
_FULL_IMAGE_MODELS = {"f3net", "vit", "hive"}


def _to_analysis_metrics(ar: AnalysisResult) -> AnalysisMetrics | None:
    """Convert an AnalysisResult dataclass into the pydantic AnalysisMetrics schema."""
    if ar is None:
        return None
    try:
        fft = FFTMetrics(
            low_freq=ar.fft_low_freq,
            mid_freq=ar.fft_mid_freq,
            high_freq=ar.fft_high_freq,
            spectral_irregularity=ar.fft_spectral_irregularity,
            profile=ar.fft_profile or [],
        )
        texture = TextureMetrics(
            sharpness=ar.sharpness,
            texture_uniformity=ar.texture_uniformity,
            noise_level=ar.noise_level,
            compression_artifacts=ar.compression_artifacts,
        )
        skin: SkinMetrics | None = None
        if ar.skin_pore_detail is not None:
            skin = SkinMetrics(
                pore_detail=ar.skin_pore_detail,
                blotchiness=ar.skin_blotchiness or 0.0,
                edge_blend=ar.skin_edge_blend or 0.0,
            )
        return AnalysisMetrics(
            fft=fft,
            texture=texture,
            symmetry_score=ar.symmetry_score,
            skin=skin,
            top_attention_regions=ar.top_attention_regions or [],
            region_scores=ar.region_scores or {},
        )
    except Exception:
        return None


class DetectionPipeline:
    def __init__(self) -> None:
        self._registry = ModelRegistry.get_instance()

    async def run(
        self,
        image: Image.Image,
        apply_display: bool = True,
        skip_models: frozenset[str] = frozenset(),
        fake_threshold: float = 0.62,
    ) -> DetectionResponse:
        """Run the full detection pipeline.

        Args:
            image: The PIL image to analyse.
            apply_display: If True (default), map raw scores to the display range
                [0.86–0.96] for fake, [0.04–0.14] for real.  Set False when the
                caller (e.g. video pipeline) wants raw probabilities for
                aggregation and will apply display mapping itself.
            skip_models: Model names to exclude from this run.  Useful for
                skipping slow or irrelevant models (e.g. "vit", "hive") in the
                video per-frame loop.
            fake_threshold: Raw score above which a prediction is considered
                "fake" for display mapping.  Default 0.62 for images; pass 0.50
                for video aggregation.
        """
        t_start = time.time()

        source_rgb = np.array(image.convert("RGB"))
        face_crop = detect_largest_face(source_rgb)
        face_detected = face_crop is not None

        full_image = image.convert("RGB")
        face_image = Image.fromarray(face_crop) if face_detected else full_image

        # Two preprocess dicts — share work where models agree on input view
        face_input = preprocess(face_image)
        full_input = preprocess(full_image)

        # Dispatch loaded detectors in parallel via a thread pool.
        # Models in skip_models are excluded (e.g. "vit"/"hive" for video frames).
        loop = asyncio.get_event_loop()
        tasks: dict[str, asyncio.Future] = {}
        for name, det in self._registry.all().items():
            if name in skip_models:
                continue
            chosen_input = full_input if name in _FULL_IMAGE_MODELS else face_input
            tasks[name] = loop.run_in_executor(None, det.predict, chosen_input)

        outputs = {name: await fut for name, fut in tasks.items()}

        if not outputs:
            raise RuntimeError("No detection models are loaded.")

        # Build model votes — hive runs silently; its signal feeds fusion but is never exposed.
        # Apply display mapping when apply_display=True.
        def _maybe_display(raw: float) -> float:
            return _display_score(raw, fake_threshold) if apply_display else raw

        model_votes: dict[str, ModelVote] = {
            name: ModelVote(
                fake_prob=_maybe_display(out.fake_prob),
                real_prob=1.0 - _maybe_display(out.fake_prob),
                inference_time_ms=out.inference_time_ms,
            )
            for name, out in outputs.items()
            if name != "hive"
        }

        # Fusion (all loaded models including hive contribute to raw final_score)
        scores = {name: out.fake_prob for name, out in outputs.items()}
        fusion = fuse(scores, face_detected=face_detected)

        # Displayed fusion weights: strip hive and re-normalise so the visible models sum to 100%
        visible_weights = {k: v for k, v in fusion.weights.items() if k != "hive"}
        _w_total = sum(visible_weights.values())
        if _w_total > 0:
            visible_weights = {k: v / _w_total for k, v in visible_weights.items()}

        raw_final = float(fusion.final_score)
        verdict: str = "fake" if raw_final >= 0.5 else "real"

        # ── Analysis suite ────────────────────────────────────────────────────
        # Try to get saliency from SigLIP first (face-focused), then ViT.
        saliency: np.ndarray | None = None
        for model_name in ("siglip", "vit"):
            det = self._registry.get(model_name)
            if det is None:
                continue
            chosen_input = full_input if model_name in _FULL_IMAGE_MODELS else face_input
            try:
                saliency = det.get_heatmap(chosen_input)
                if saliency is not None:
                    break
            except Exception:
                pass

        analysis_result: AnalysisResult | None = None
        try:
            analysis_result = run_analysis(full_image, saliency=saliency)
        except Exception:
            pass

        analysis_metrics = _to_analysis_metrics(analysis_result) if analysis_result else None

        explanations = self._build_explanations(
            scores=scores,
            face_detected=face_detected,
            is_uncertain=fusion.is_uncertain,
            final_score=fusion.final_score,
            analysis=analysis_result,
        )

        total_ms = (time.time() - t_start) * 1000

        return DetectionResponse(
            result_id=str(uuid.uuid4()),
            final_score=_maybe_display(raw_final),
            verdict=verdict,  # type: ignore[arg-type]
            face_detected=face_detected,
            is_uncertain=fusion.is_uncertain,
            model_votes=model_votes,
            fusion_weights=visible_weights,
            explanations=explanations,
            total_inference_time_ms=total_ms,
            analysis=analysis_metrics,
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
        analysis: AnalysisResult | None = None,
    ) -> list[str]:
        """
        Produces a short, ordered list of human-readable findings.
        Each line is independent — the frontend renders them as bullets.
        """
        explanations: list[str] = []

        # ── Top-line verdict statement ────────────────────────────────────────
        pct = round(final_score * 100)
        if final_score >= 0.5:
            explanations.append(
                f"Overall, this image is {pct}% likely to be a deepfake. "
                f"Multiple detectors agree on synthesis artifacts."
                if pct >= 70
                else f"Overall, this image leans fake ({pct}% confidence) but signals are mixed."
            )
        else:
            real_pct = 100 - pct
            explanations.append(
                f"Overall, this image looks authentic ({real_pct}% confidence). "
                f"No strong manipulation patterns detected."
                if real_pct >= 70
                else f"Overall, this image leans real ({real_pct}% confidence) but signals are mixed."
            )

        # ── Face-detection context ────────────────────────────────────────────
        if not face_detected:
            explanations.append(
                "No face was detected, so face-cropped models analysed the full image instead."
            )

        # ── Per-model findings, sorted strongest signal first ─────────────────
        eff = scores.get("efficientnet")
        f3 = scores.get("f3net")
        vit = scores.get("vit")
        xcep = scores.get("xceptionnet")
        siglip = scores.get("siglip")
        # hive score is intentionally not surfaced in explanations

        # FAKE-direction model findings
        if vit is not None and vit >= 0.6:
            explanations.append(
                f"ViT (full-image transformer) is {round(vit * 100)}% confident the image is synthetic — "
                "it spotted learned generation patterns across the whole frame."
            )
        if siglip is not None and siglip >= 0.6:
            explanations.append(
                f"SigLIP (vision-language model) is {round(siglip * 100)}% confident it's fake — "
                "cross-modal semantic features indicate AI-generated facial characteristics."
            )
        if f3 is not None and f3 >= 0.6:
            explanations.append(
                f"F3Net (frequency analyser) is {round(f3 * 100)}% confident it's fake — "
                "the DCT decomposition revealed unnatural frequency-band energy typical of GAN/diffusion synthesis."
            )
        if eff is not None and eff >= 0.6:
            explanations.append(
                f"EfficientNet (facial-texture CNN) is {round(eff * 100)}% confident it's fake — "
                "it found micro-texture inconsistencies in the facial region (skin pores, edge blending, eye reflections)."
            )
        if xcep is not None and xcep >= 0.6:
            explanations.append(
                f"XceptionNet flagged localised manipulation artifacts ({round(xcep * 100)}% fake)."
            )

        # REAL-direction model findings — only mention strong real signals
        if vit is not None and vit <= 0.2:
            explanations.append(
                f"ViT is {round((1 - vit) * 100)}% confident the image is authentic — "
                "no synthesis patterns visible in the global image structure."
            )
        if siglip is not None and siglip <= 0.2:
            explanations.append(
                f"SigLIP is {round((1 - siglip) * 100)}% confident the image is real — "
                "vision-language features match authentic facial imagery."
            )
        if f3 is not None and f3 <= 0.2:
            explanations.append(
                f"F3Net is {round((1 - f3) * 100)}% confident the image is real — "
                "frequency-domain energy matches that of natural photographs."
            )

        # ── Analysis-suite signals ────────────────────────────────────────────
        if analysis is not None:
            if analysis.fft_spectral_irregularity > 0.6:
                explanations.append(
                    f"FFT analysis detected elevated spectral irregularity ({analysis.fft_spectral_irregularity:.2f}) — "
                    "unnatural high-frequency energy is a hallmark of AI-generated images."
                )
            if analysis.symmetry_score is not None and analysis.symmetry_score < 0.6:
                explanations.append(
                    f"Facial symmetry analysis returned a low score ({analysis.symmetry_score:.2f}) — "
                    "real faces tend to be more symmetrical; this may indicate compositing artifacts."
                )
            if analysis.skin_pore_detail is not None and analysis.skin_pore_detail < 0.3:
                explanations.append(
                    "Skin quality analysis found unusually smooth texture — "
                    "diffusion models often over-smooth fine skin details like pores."
                )
            if analysis.top_attention_regions:
                regions = ", ".join(analysis.top_attention_regions[:3])
                explanations.append(
                    f"Attention analysis highlighted these regions as most suspicious: {regions}."
                )

        # ── Uncertainty warning ───────────────────────────────────────────────
        if is_uncertain:
            explanations.append(
                f"⚠ The final score ({final_score:.2f}) falls inside the uncertainty band (0.38–0.62). "
                "Treat this verdict as low confidence — borderline cases benefit from human review."
            )

        return explanations
