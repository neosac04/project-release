"""Texture and compression-artefact metrics — adapted from deepfake-detector-model-v1/app.py."""
from __future__ import annotations

import numpy as np
from PIL import Image


def texture_metrics(img_pil: Image.Image) -> dict:
    try:
        import cv2
    except ImportError:
        return {"sharpness": 0.0, "texture_uniformity": 0.0, "noise_level": 0.0, "compression_artifacts": 0.0}

    arr  = np.array(img_pil.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    k5 = np.ones((5, 5), np.float32) / 25
    mean_f   = cv2.filter2D(gray.astype(np.float32), -1, k5)
    sq_mean  = cv2.filter2D(gray.astype(np.float32) ** 2, -1, k5)
    local_std = np.sqrt(np.maximum(sq_mean - mean_f ** 2, 0))

    hp = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32) / 8
    noise_map = cv2.filter2D(gray.astype(np.float32), -1, hp)

    h, w = gray.shape
    bdiffs: list[float] = (
        [float(np.abs(gray[i * 8, :].astype(float) - gray[i * 8 - 1, :].astype(float)).mean())
         for i in range(1, h // 8)]
        + [float(np.abs(gray[:, j * 8].astype(float) - gray[:, j * 8 - 1].astype(float)).mean())
           for j in range(1, w // 8)]
    )

    return {
        "sharpness":             round(min(lap_var / 1000.0, 1.0), 4),
        "texture_uniformity":    round(min(float(np.std(local_std)) / 50.0, 1.0), 4),
        "noise_level":           round(min(float(np.std(noise_map)) / 20.0, 1.0), 4),
        "compression_artifacts": round(min(float(np.mean(bdiffs)) * 100 if bdiffs else 0, 1.0), 4),
    }
