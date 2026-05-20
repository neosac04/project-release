"""Analysis suite orchestrator — FFT, texture, symmetry, region attention, skin quality."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from app.analysis.face_analysis import get_landmarks, region_attention, skin_quality, symmetry_score
from app.analysis.fft_analysis import frequency_analysis
from app.analysis.texture_analysis import texture_metrics


@dataclass
class AnalysisResult:
    fft_low_freq: float = 0.0
    fft_mid_freq: float = 0.0
    fft_high_freq: float = 0.0
    fft_spectral_irregularity: float = 0.0
    fft_profile: list[float] = field(default_factory=list)

    sharpness: float = 0.0
    texture_uniformity: float = 0.0
    noise_level: float = 0.0
    compression_artifacts: float = 0.0

    symmetry_score: float | None = None
    top_attention_regions: list[str] = field(default_factory=list)
    region_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    skin_pore_detail: float | None = None
    skin_blotchiness: float | None = None
    skin_edge_blend: float | None = None


def run_analysis(img_pil: Image.Image, saliency: np.ndarray | None = None) -> AnalysisResult:
    result = AnalysisResult()

    try:
        fft = frequency_analysis(img_pil)
        result.fft_low_freq              = fft["low_freq"]
        result.fft_mid_freq              = fft["mid_freq"]
        result.fft_high_freq             = fft["high_freq"]
        result.fft_spectral_irregularity = fft["spectral_irregularity"]
        result.fft_profile               = fft["profile"]
    except Exception as exc:
        print(f"⚠️ FFT analysis failed: {exc!r}")

    try:
        tex = texture_metrics(img_pil)
        result.sharpness             = tex["sharpness"]
        result.texture_uniformity    = tex["texture_uniformity"]
        result.noise_level           = tex["noise_level"]
        result.compression_artifacts = tex["compression_artifacts"]
    except Exception as exc:
        print(f"⚠️ Texture analysis failed: {exc!r}")

    try:
        lm_pts = get_landmarks(img_pil)
        result.symmetry_score = symmetry_score(lm_pts, img_pil)

        if lm_pts is not None:
            regions = region_attention(img_pil, lm_pts, saliency)
            result.region_scores = regions
            if regions:
                result.top_attention_regions = sorted(
                    regions, key=lambda r: regions[r]["attention"], reverse=True
                )[:3]

            skin = skin_quality(img_pil, lm_pts)
            result.skin_pore_detail = skin["pore_detail"]
            result.skin_blotchiness = skin["blotchiness"]
            result.skin_edge_blend  = skin["edge_blend"]
    except Exception as exc:
        print(f"⚠️ Face analysis failed: {exc!r}")

    return result
