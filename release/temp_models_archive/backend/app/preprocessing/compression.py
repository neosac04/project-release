from __future__ import annotations
import base64
import io
import numpy as np
from PIL import Image
from app.schemas.response import CompressionAnalysis

try:
    import cv2
except Exception:
    cv2 = None


class CompressionAnalyzer:
    """
    JPEG compression artifact analysis.
    Error Level Analysis (ELA): regions recompressed differently from authentic baseline.
    Block artifact detection: 8×8 DCT boundary inconsistencies from double compression.
    """

    def analyze(self, image: Image.Image) -> CompressionAnalysis:
        return CompressionAnalysis(
            ela_score=self._ela_score(image),
            block_artifact_score=self._block_artifacts(image),
            ela_image_b64=self._ela_visualization(image),
        )

    def _ela_score(self, image: Image.Image) -> float:
        """
        ELA: save at known quality, compare pixel-level differences.
        - Authentic regions: low ELA (already at equilibrium).
        - Edited/synthetic regions: high ELA (recompress differently).
        - AI-generated: very uniform low ELA (suspicious uniformity).
        Returns the standard deviation of the ELA map (higher = more spatial variance).
        """
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")

        orig = np.array(image.convert("RGB")).astype(np.float32)
        recomp = np.array(recompressed).astype(np.float32)

        ela = np.abs(orig - recomp)
        ela_scaled = ela * 10.0
        ela_norm = ela_scaled / 255.0

        std = float(ela_norm.std())
        # AI images: very low std (~0.01-0.05); edited images: moderate-high std (>0.1)
        return float(np.clip(std, 0.0, 1.0))

    def _block_artifacts(self, image: Image.Image) -> float:
        """
        JPEG 8×8 DCT block boundaries.
        Double-compressed or manipulated images show discontinuities at block edges.
        Higher score = stronger block artifacts = more suspicious of manipulation.
        """
        gray = np.array(image.convert("L")).astype(np.float64)
        h, w = gray.shape

        v_boundary = 0.0
        v_count = 0
        for i in range(8, w, 8):
            v_boundary += float(np.abs(gray[:, i] - gray[:, i - 1]).mean())
            v_count += 1

        h_boundary = 0.0
        h_count = 0
        for i in range(8, h, 8):
            h_boundary += float(np.abs(gray[i, :] - gray[i - 1, :]).mean())
            h_count += 1

        total_v_grad = float(np.abs(np.diff(gray, axis=1)).mean())
        total_h_grad = float(np.abs(np.diff(gray, axis=0)).mean())

        v_ratio = (v_boundary / v_count) / (total_v_grad + 1e-8) if v_count > 0 else 1.0
        h_ratio = (h_boundary / h_count) / (total_h_grad + 1e-8) if h_count > 0 else 1.0

        artifact = (v_ratio + h_ratio) / 2.0
        # Ratio ~1 is normal, >1.3 suggests double compression
        return float(np.clip((artifact - 1.0) / 0.5, 0.0, 1.0))

    def _ela_visualization(self, image: Image.Image) -> str:
        """Returns base64-encoded amplified ELA difference image."""
        if cv2 is None:
            return ""

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")

        orig = np.array(image.convert("RGB")).astype(np.float32)
        recomp = np.array(recompressed).astype(np.float32)

        ela = np.abs(orig - recomp) * 10.0
        ela = np.clip(ela, 0, 255).astype(np.uint8)
        ela_gray = ela.mean(axis=2).astype(np.uint8)
        colored = cv2.applyColorMap(ela_gray, cv2.COLORMAP_HOT)
        colored = cv2.resize(colored, (256, 256))
        _, buf2 = cv2.imencode(".png", colored)
        return base64.b64encode(buf2.tobytes()).decode("utf-8")
