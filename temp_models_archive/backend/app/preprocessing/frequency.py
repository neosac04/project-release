from __future__ import annotations
import base64
import io
import numpy as np
from PIL import Image
from app.schemas.response import FrequencyAnalysis

try:
    import cv2
except Exception:
    cv2 = None


class FrequencyAnalyzer:
    """
    Frequency domain analysis for deepfake detection.
    GAN upsampling leaves characteristic spectral artifacts.
    Diffusion models over-smooth high frequencies.
    """

    def analyze(self, image: Image.Image) -> FrequencyAnalysis:
        gray = np.array(image.convert("L")).astype(np.float64)
        rgb  = np.array(image.convert("RGB")).astype(np.float64)

        return FrequencyAnalysis(
            fft_anomaly_score=self._fft_anomaly(gray),
            dct_anomaly_score=self._dct_anomaly(rgb),
            high_freq_ratio=self._high_freq_ratio(gray),
            spectral_entropy=self._spectral_entropy(gray),
            fft_image_b64=self._fft_visualization(gray),
        )

    def _fft_anomaly(self, gray: np.ndarray) -> float:
        """
        GANs produce grid-like artifacts in frequency domain.
        Elevated energy at specific spatial frequencies vs. surrounding area.
        """
        f = np.fft.fftshift(np.fft.fft2(gray))
        magnitude = np.log(np.abs(f) + 1.0)
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Inner region (low frequencies around DC)
        r_inner = min(h, w) // 8
        y_idx, x_idx = np.ogrid[:h, :w]
        dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)

        inner_mask = dist < r_inner
        outer_mask = (dist >= r_inner) & (dist < min(h, w) // 4)

        inner_mean = magnitude[inner_mask].mean() if inner_mask.any() else 0.0
        outer_mean = magnitude[outer_mask].mean() if outer_mask.any() else 1.0

        # GAN artifact: sharp ratio between inner/outer energy
        ratio = inner_mean / (outer_mean + 1e-8)
        # Real ~2-4; GAN ~6+
        score = np.clip((ratio - 2.0) / 8.0, 0.0, 1.0)
        return float(score)

    def _dct_anomaly(self, rgb: np.ndarray) -> float:
        """DCT energy distribution anomaly across channels."""
        from scipy.fftpack import dct
        scores = []
        for c in range(3):
            ch = rgb[:, :, c]
            d = dct(dct(ch, axis=0, norm="ortho"), axis=1, norm="ortho")
            d_abs = np.abs(d)
            total = d_abs.sum() + 1e-8
            # Ratio of DC component to total energy
            dc_ratio = d_abs[0, 0] / total
            scores.append(dc_ratio)
        # Very high DC ratio = over-smooth = synthetic signal
        mean_dc = float(np.mean(scores))
        return float(np.clip((mean_dc - 0.3) / 0.5, 0.0, 1.0))

    def _high_freq_ratio(self, gray: np.ndarray) -> float:
        """
        Ratio of high-frequency energy to total energy.
        Diffusion models suppress high frequencies (lower ratio).
        """
        f = np.fft.fftshift(np.fft.fft2(gray))
        magnitude = np.abs(f)
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        y_idx, x_idx = np.ogrid[:h, :w]
        dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
        r_low = min(h, w) // 4

        low_energy  = magnitude[dist < r_low].sum()
        total_energy = magnitude.sum() + 1e-8
        high_energy  = total_energy - low_energy
        return float(high_energy / total_energy)

    def _spectral_entropy(self, gray: np.ndarray) -> float:
        """
        Measure of spectral complexity.
        Synthetic images often have lower spectral entropy (less complex spectra).
        """
        f = np.fft.fft2(gray)
        power = np.abs(f) ** 2
        power_norm = power / (power.sum() + 1e-8)
        power_norm = power_norm[power_norm > 1e-12]
        entropy = -np.sum(power_norm * np.log(power_norm))
        # Normalize to ~0-1 (typical range: 5-15 for natural images)
        return float(np.clip(entropy / 20.0, 0.0, 1.0))

    def _fft_visualization(self, gray: np.ndarray) -> str:
        """Returns base64-encoded INFERNO colormap FFT magnitude spectrum."""
        if cv2 is None:
            return ""

        f = np.fft.fftshift(np.fft.fft2(gray))
        magnitude = np.log(np.abs(f) + 1.0)
        norm = ((magnitude - magnitude.min()) /
                (magnitude.max() - magnitude.min() + 1e-8) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        # Resize to 256×256 for frontend display
        colored = cv2.resize(colored, (256, 256))
        _, buf = cv2.imencode(".png", colored)
        return base64.b64encode(buf.tobytes()).decode("utf-8")
