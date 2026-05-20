"""
Facial landmark-based analysis — symmetry, region attention, skin quality.
Adapted from deepfake-detector-model-v1/app.py.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

REGIONS: dict[str, list[int]] = {
    "left_eye":    [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "right_eye":   [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    "nose":        [1, 2, 5, 4, 19, 94, 125, 141, 235, 240, 97, 98, 99, 129],
    "mouth":       [13, 14, 17, 18, 84, 85, 86, 87, 88, 91, 95, 146, 178, 179, 180, 181, 182, 183],
    "left_cheek":  [116, 117, 118, 119, 120, 121, 126, 142, 203, 206, 207],
    "right_cheek": [345, 346, 347, 348, 349, 350, 355, 371, 423, 426, 427],
    "forehead":    [10, 21, 54, 67, 103, 104, 105, 107, 108, 109, 151],
    "jaw":         [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379],
}

_SYM_PAIRS = [(33, 263), (133, 362), (70, 300), (105, 334), (61, 291), (234, 454), (116, 345)]


def get_landmarks(img_pil: Image.Image) -> np.ndarray | None:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as _mp_py
        from mediapipe.tasks.python import vision as _mp_vis
        from pathlib import Path

        # __file__ = .../backend/app/analysis/face_analysis.py → parents[2] = .../backend
        backend_root = Path(__file__).resolve().parents[2]
        task_path = str(backend_root / "app" / "models" / "mediapipe" / "face_landmarker.task")

        opts = _mp_vis.FaceLandmarkerOptions(
            base_options=_mp_py.BaseOptions(model_asset_path=task_path),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        arr = np.array(img_pil.convert("RGB"))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)

        with _mp_vis.FaceLandmarker.create_from_options(opts) as det:
            result = det.detect(mp_img)

        if not result.face_landmarks:
            return None

        h, w = arr.shape[:2]
        return np.array(
            [[lm.x * w, lm.y * h] for lm in result.face_landmarks[0]],
            dtype=np.float32,
        )
    except Exception as exc:
        print(f"⚠️ Landmark detection failed: {exc!r}")
        return None


def symmetry_score(lm_pts: np.ndarray | None, img_pil: Image.Image) -> float | None:
    if lm_pts is None or len(lm_pts) < 2:
        return None
    w, h = img_pil.size
    nose_x = float(lm_pts[1, 0])
    diffs: list[float] = []
    for l_idx, r_idx in _SYM_PAIRS:
        if l_idx >= len(lm_pts) or r_idx >= len(lm_pts):
            continue
        lx, ly = lm_pts[l_idx]
        rx, ry = lm_pts[r_idx]
        diffs.append(
            abs(abs(lx - nose_x) - abs(rx - nose_x)) / w
            + abs(ly - ry) / h * 0.5
        )
    if not diffs:
        return None
    return round(max(0.0, 1.0 - float(np.mean(diffs)) * 10), 4)


def region_attention(
    img_pil: Image.Image,
    lm_pts: np.ndarray | None,
    saliency: np.ndarray | None,
) -> dict[str, dict[str, float]]:
    if lm_pts is None or saliency is None:
        return {}
    try:
        import cv2
    except ImportError:
        return {}

    arr = np.array(img_pil.convert("RGB"))
    h, w = arr.shape[:2]
    sal_r = cv2.resize(saliency, (w, h))

    scores: dict[str, dict[str, float]] = {}
    for name, idxs in REGIONS.items():
        valid = [i for i in idxs if i < len(lm_pts)]
        if len(valid) < 3:
            continue
        pts = lm_pts[valid].astype(np.int32)
        pts[:, 0] = pts[:, 0].clip(0, w - 1)
        pts[:, 1] = pts[:, 1].clip(0, h - 1)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts, 255)
        region_sal = sal_r[mask > 0]
        region_px  = arr[mask > 0]
        if len(region_sal) == 0:
            continue
        scores[name] = {
            "attention": round(float(region_sal.mean()), 4),
            "texture":   round(float(np.var(region_px.astype(float)) / 65025), 4),
        }
    return scores


def skin_quality(img_pil: Image.Image, lm_pts: np.ndarray | None) -> dict[str, float]:
    empty = {"pore_detail": 0.0, "blotchiness": 0.0, "edge_blend": 0.0}
    if lm_pts is None:
        return empty
    try:
        import cv2
    except ImportError:
        return empty

    arr  = np.array(img_pil.convert("RGB"))
    h, w = arr.shape[:2]
    cheek_idxs = REGIONS["left_cheek"] + REGIONS["right_cheek"]
    valid = [i for i in cheek_idxs if i < len(lm_pts)]
    if len(valid) < 3:
        return empty

    pts = lm_pts[valid].astype(np.int32)
    pts[:, 0] = pts[:, 0].clip(0, w - 1)
    pts[:, 1] = pts[:, 1].clip(0, h - 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if len(gray[mask > 0]) < 10:
        return empty

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    pore_detail = round(min(float(np.std(lap[mask > 0])) / 30.0, 1.0), 4)

    skin_rgb = arr[mask > 0].astype(float)
    blotch   = round(min(float(np.std(skin_rgb)) / 60.0, 1.0), 4)

    dx  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gmag = np.sqrt(dx ** 2 + dy ** 2)
    dil  = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    ring = dil ^ mask
    edge_val   = float(gmag[ring > 0].mean()) if ring.any() else 0.0
    edge_blend = round(1.0 - min(edge_val / 60.0, 1.0), 4)

    return {"pore_detail": pore_detail, "blotchiness": blotch, "edge_blend": edge_blend}
