from __future__ import annotations
import base64
import numpy as np
from PIL import Image
from app.schemas.response import PRNUAnalysis

try:
    import cv2
except Exception:
    cv2 = None

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    def gaussian_filter(array, sigma=2.0):
        array = np.asarray(array, dtype=np.float64)
        radius = max(1, int(round(float(sigma) * 2)))
        size = radius * 2 + 1
        kernel_1d = np.exp(-0.5 * (np.arange(size) - radius) ** 2 / max(float(sigma) ** 2, 1e-8))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        padded = np.pad(array, radius, mode="reflect")
        result = np.zeros_like(array, dtype=np.float64)
        for y in range(array.shape[0]):
            for x in range(array.shape[1]):
                window = padded[y:y + size, x:x + size]
                result[y, x] = float(np.sum(window * kernel_2d))
        return result


class PRNUAnalyzer:
    """
    Photo Response Non-Uniformity (PRNU) analysis.

    Every real camera sensor leaves a unique noise fingerprint.
    AI-generated images lack this fingerprint entirely — a strong
    positive signal for authenticity when the fingerprint IS present.

    Pipeline:
      1. Extract noise residual: N = I - gaussian_filter(I)
      2. Measure autocorrelation structure of N
      3. Detect GAN periodic patterns in N via FFT
    """

    def analyze(self, image: Image.Image) -> PRNUAnalysis:
        img = np.array(image.convert("RGB")).astype(np.float64)
        noise = self._extract_noise(img)

        return PRNUAnalysis(
            prnu_correlation=self._autocorrelation_score(noise),
            noise_pattern_score=self._periodic_pattern_score(noise),
            prnu_map_b64=self._visualize_noise(noise),
        )

    def _extract_noise(self, img: np.ndarray) -> np.ndarray:
        """N = I - F(I), where F is per-channel Gaussian denoising."""
        noise = np.zeros_like(img)
        for c in range(3):
            smoothed = gaussian_filter(img[:, :, c], sigma=2.0)
            noise[:, :, c] = img[:, :, c] - smoothed
        return noise

    def _autocorrelation_score(self, noise: np.ndarray) -> float:
        """
        Measure off-center autocorrelation of noise residual.
        Real cameras: spatially correlated PRNU (score ~0.3-0.7).
        Synthetic: random noise (score ~0.05-0.2).
        Returns 0-1 where higher = more camera-like = more likely real.
        """
        gray = noise.mean(axis=2)
        N = gray - gray.mean()
        variance = (N ** 2).sum()
        if variance < 1e-8:
            return 0.5

        corr_sum = 0.0
        offsets = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4)
                   if not (dy == 0 and dx == 0)]

        for dy, dx in offsets:
            shifted = np.roll(np.roll(N, dy, axis=0), dx, axis=1)
            corr_sum += abs((N * shifted).sum() / variance)

        normalized = corr_sum / len(offsets)
        # Scale: real ~0.15-0.4, fake ~0.02-0.1
        score = np.clip((normalized - 0.02) / 0.38, 0.0, 1.0)
        return float(score)

    def _periodic_pattern_score(self, noise: np.ndarray) -> float:
        """
        GAN artifacts manifest as periodic structures in the noise residual
        (regular peaks in the FFT of the noise).
        Low score = many periodic peaks = GAN-like = less authentic.
        """
        gray = noise.mean(axis=2)
        fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
        fft_log = np.log(fft_mag + 1.0)
        fft_norm = fft_log / (fft_log.max() + 1e-8)

        h, w = fft_norm.shape
        cy, cx = h // 2, w // 2
        # Remove DC component
        fft_no_dc = fft_norm.copy()
        fft_no_dc[cy - 5:cy + 5, cx - 5:cx + 5] = 0.0

        threshold = fft_no_dc.mean() + 3.0 * fft_no_dc.std()
        peak_density = (fft_no_dc > threshold).sum() / fft_no_dc.size
        # High peak density = GAN periodic artifact = lower authenticity
        score = max(0.0, 1.0 - peak_density * 500)
        return float(np.clip(score, 0.0, 1.0))

    def _visualize_noise(self, noise: np.ndarray) -> str:
        """Visualize PRNU noise residual as a base64 PNG."""
        if cv2 is None:
            return ""

        gray_noise = noise.mean(axis=2)
        # Amplify and normalize for visibility
        amplified = gray_noise * 10.0
        normalized = np.clip(amplified + 128, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_MAGMA)
        colored = cv2.resize(colored, (256, 256))
        _, buf = cv2.imencode(".png", colored)
        return base64.b64encode(buf.tobytes()).decode("utf-8")
