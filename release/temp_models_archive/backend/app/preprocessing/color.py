from __future__ import annotations
import numpy as np
from PIL import Image
from app.schemas.response import ColorAnalysis


class ColorAnalyzer:
    """
    RGB channel statistics analysis.
    Synthetic images often exhibit unnatural inter-channel correlations
    and atypical histogram distributions.
    """

    def analyze(self, image: Image.Image) -> ColorAnalysis:
        img = np.array(image.convert("RGB")).astype(np.float64)
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        return ColorAnalysis(
            r_mean=float(r.mean()),
            r_std=float(r.std()),
            g_mean=float(g.mean()),
            g_std=float(g.std()),
            b_mean=float(b.mean()),
            b_std=float(b.std()),
            channel_correlation_score=self._channel_correlation(r, g, b),
            histogram_uniformity=self._histogram_uniformity(img),
        )

    def _channel_correlation(self, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> float:
        """
        Natural photos have specific R-G-B correlation patterns from scene lighting.
        GAN images often show unnaturally high or uniform correlations.
        Returns 0-1: values near 0.5 are natural; extremes are suspicious.
        """
        flat_r = r.flatten()
        flat_g = g.flatten()
        flat_b = b.flatten()

        def pearson(a: np.ndarray, x: np.ndarray) -> float:
            cov = np.cov(a, x)
            denom = np.sqrt(cov[0, 0] * cov[1, 1]) + 1e-8
            return abs(cov[0, 1] / denom)

        rg = pearson(flat_r, flat_g)
        rb = pearson(flat_r, flat_b)
        gb = pearson(flat_g, flat_b)
        mean_corr = (rg + rb + gb) / 3.0

        # Very high correlation (>0.98) suggests synthetic uniform generation
        anomaly = max(0.0, (mean_corr - 0.85) / 0.15)
        return float(np.clip(anomaly, 0.0, 1.0))

    def _histogram_uniformity(self, img: np.ndarray) -> float:
        """
        Measures how uniform the histogram is across bins.
        Real photos: varied histogram (low uniformity in specific ways).
        Some GAN outputs: unusually smooth/uniform histograms.
        Returns chi-squared distance from a flat histogram (higher = less uniform).
        """
        scores = []
        for c in range(3):
            hist, _ = np.histogram(img[:, :, c].flatten(), bins=64, range=(0, 256))
            hist = hist.astype(np.float64)
            expected = hist.sum() / 64.0
            chi2 = ((hist - expected) ** 2 / (expected + 1e-8)).sum()
            # Normalize to 0-1
            scores.append(np.clip(chi2 / 5000.0, 0.0, 1.0))
        return float(np.mean(scores))
