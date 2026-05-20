from __future__ import annotations
import base64
import io
import numpy as np
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None


def heatmap_to_overlay(
    original: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> bytes:
    """
    Resize heatmap (H×W float32, range 0-1) to match original image,
    apply JET colormap (blue=real, red=fake), blend at alpha, return PNG bytes.
    """
    if cv2 is None:
        buf = io.BytesIO()
        original.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()

    w, h = original.size
    hm_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    hm_uint8 = (np.clip(hm_resized, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)           # BGR
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    orig_np = np.array(original.convert("RGB")).astype(np.float32)
    blended = (orig_np * (1 - alpha) + colored_rgb.astype(np.float32) * alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return buf.getvalue()


def ensemble_heatmap(heatmaps: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """
    Weighted average of multiple heatmaps. Heatmaps from different model
    architectures may have different spatial resolutions, so we upsample all to the
    largest grid before combining.
    """
    if not heatmaps:
        return np.zeros((14, 14), dtype=np.float32)

    target_h = max(hm.shape[0] for hm in heatmaps)
    target_w = max(hm.shape[1] for hm in heatmaps)

    resized: list[np.ndarray] = []
    for hm in heatmaps:
        if hm.shape == (target_h, target_w):
            resized.append(hm.astype(np.float32))
        elif cv2 is not None:
            resized.append(
                cv2.resize(hm.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            )
        else:
            # Fallback: nearest-neighbour upsample with numpy
            ratio_h = target_h / hm.shape[0]
            ratio_w = target_w / hm.shape[1]
            idx_h = (np.arange(target_h) / ratio_h).astype(int)
            idx_w = (np.arange(target_w) / ratio_w).astype(int)
            resized.append(hm[idx_h[:, None], idx_w[None, :]].astype(np.float32))

    total_w = sum(weights) or 1.0
    result = np.zeros((target_h, target_w), dtype=np.float32)
    for hm, w in zip(resized, weights):
        result += hm * (w / total_w)
    mn, mx = float(result.min()), float(result.max())
    return ((result - mn) / (mx - mn + 1e-8)).astype(np.float32)
