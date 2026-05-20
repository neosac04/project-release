"""FFT frequency analysis — adapted from deepfake-detector-model-v1/app.py."""
from __future__ import annotations

import numpy as np
from PIL import Image


def frequency_analysis(img_pil: Image.Image) -> dict:
    gray = np.array(img_pil.convert("L")).astype(np.float32)
    fft  = np.fft.fftshift(np.fft.fft2(gray))
    mag  = np.log(np.abs(fft) + 1.0)
    mag_n = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

    h, w  = mag_n.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    dist   = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    md     = float(np.sqrt(cy ** 2 + cx ** 2))

    N_BINS = 64
    profile: list[float] = []
    for i in range(N_BINS):
        band = (dist >= md * i / N_BINS) & (dist < md * (i + 1) / N_BINS)
        profile.append(float(mag_n[band].mean()) if band.any() else 0.0)

    return {
        "low_freq":              round(float(mag_n[dist < md * 0.1].mean()), 4),
        "mid_freq":              round(float(mag_n[(dist >= md * 0.1) & (dist < md * 0.5)].mean()), 4),
        "high_freq":             round(float(mag_n[dist >= md * 0.5].mean()), 4),
        "spectral_irregularity": round(float(np.std(profile)), 4),
        "profile":               [round(v, 4) for v in profile],
    }
