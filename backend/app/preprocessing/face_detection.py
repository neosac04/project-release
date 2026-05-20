from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def _to_rgb(image: np.ndarray, bgr_input: bool = False) -> np.ndarray:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image input")
    if cv2 is None:
        raise RuntimeError("OpenCV is unavailable")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] >= 3:
        if bgr_input:
            return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        return image[:, :, :3].copy()  # already RGB (PIL-derived)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _pick_largest_bbox(boxes: list[list[float]]) -> list[float] | None:
    if not boxes:
        return None
    return max(boxes, key=lambda b: max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))


@lru_cache(maxsize=1)
def _get_mediapipe_detector():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        repo_root = Path(__file__).resolve().parents[3]
        model_path = str(repo_root / "backend" / "app" / "models" / "mediapipe" / "face_landmarker.task")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=5,
        )
        return vision.FaceLandmarker.create_from_options(options)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _get_haar_cascade():
    if cv2 is None:
        return None
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    return None if cascade.empty() else cascade


def _mediapipe_boxes(rgb_image: np.ndarray) -> list[list[float]]:
    detector = _get_mediapipe_detector()
    if detector is None:
        return []
    try:
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = detector.detect(mp_image)
        boxes: list[list[float]] = []
        height, width = rgb_image.shape[:2]
        for landmarks in (result.face_landmarks or []):
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            if not xs or not ys:
                continue
            x1 = max(0.0, min(xs)) * width
            y1 = max(0.0, min(ys)) * height
            x2 = min(1.0, max(xs)) * width
            y2 = min(1.0, max(ys)) * height
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
        return boxes
    except Exception:
        return []


def _haar_boxes(rgb_image: np.ndarray) -> list[list[float]]:
    cascade = _get_haar_cascade()
    if cascade is None or cv2 is None:
        return []
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    detections = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return [[float(x), float(y), float(x + w), float(y + h)] for x, y, w, h in detections]


def detect_largest_face(image: np.ndarray, bgr_input: bool = False) -> Optional[np.ndarray]:
    try:
        rgb = _to_rgb(image, bgr_input=bgr_input)
    except Exception:
        return None

    boxes = _mediapipe_boxes(rgb)
    if not boxes:
        boxes = _haar_boxes(rgb)

    largest = _pick_largest_bbox(boxes)
    if largest is None:
        return None

    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = largest
    x1 = max(0, int(np.floor(x1)))
    y1 = max(0, int(np.floor(y1)))
    x2 = min(w, int(np.ceil(x2)))
    y2 = min(h, int(np.ceil(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return rgb[y1:y2, x1:x2].copy()


def test_face_detection(image_path: str):
    """
    Loads image, runs face detection, and displays:
    - original image
    - cropped face (if found)
    """
    import matplotlib.pyplot as plt

    if cv2 is None:
        raise RuntimeError("OpenCV is unavailable")

    image = cv2.imread(image_path)
    if image is None:
        print("No face detected")
        return None

    face = detect_largest_face(image, bgr_input=True)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if face is None:
        print("No face detected")
        blank = np.zeros((10, 10, 3), dtype=np.uint8)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(rgb)
        plt.title("Original Image")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(blank)
        plt.title("Cropped Face")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        return None

    print("Face detected")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(face)
    plt.title("Cropped Face")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return face