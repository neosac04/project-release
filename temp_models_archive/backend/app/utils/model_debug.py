from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

from app.preprocessing.face_detection import detect_largest_face


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_clip_transform = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(CLIP_MEAN, CLIP_STD),
])

_face_transform = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _normalize_device(device: str | torch.device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _as_rgb_image(image: np.ndarray) -> np.ndarray:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image input")

    if cv2 is None:
        raise RuntimeError("OpenCV is unavailable")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[2] < 3:
        raise ValueError(f"Unsupported channel count: {image.shape[2]}")

    return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)


def _to_pil_rgb(image: np.ndarray) -> Image.Image:
    return Image.fromarray(_as_rgb_image(image))


def _ensure_batch(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.unsqueeze(0) if tensor.ndim == 3 else tensor


def _resolve_torch_model(model: Any) -> Any:
    if hasattr(model, "model") and getattr(model, "model") is not None:
        return model.model
    return model


def _move_model_to_device(model: Any, device: torch.device) -> Any:
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model


def _move_univfd_to_device(model: Any, device: torch.device) -> Any:
    if hasattr(model, "clip_model") and model.clip_model is not None:
        model.clip_model = model.clip_model.to(device).eval()
    if hasattr(model, "linear") and model.linear is not None:
        model.linear = model.linear.to(device).eval()
    if hasattr(model, "device"):
        model.device = device
    return model


def _sigmoid_probability_from_logits(logits: torch.Tensor) -> float:
    logits = logits.float()
    if logits.ndim == 0:
        return float(torch.sigmoid(logits).item())

    flat = logits.reshape(-1)
    if flat.numel() == 1:
        return float(torch.sigmoid(flat[0]).item())

    if flat.numel() >= 2:
        return float(torch.sigmoid(flat[1] - flat[0]).item())

    return float(torch.sigmoid(flat.squeeze()).item())


def _collect_image_paths(directory: str | Path, limit: int | None = None) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        return []

    paths = [
        path for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in _SUPPORTED_IMAGE_SUFFIXES
    ]
    if limit is not None:
        return paths[:limit]
    return paths


def preprocess_univfd(image: np.ndarray) -> torch.Tensor:
    """Prepare an image for the UnivFD CLIP backbone."""
    pil_image = _to_pil_rgb(image)
    tensor = _clip_transform(pil_image)
    return _ensure_batch(tensor)


def preprocess_face_model(image: np.ndarray, size: int = 224) -> torch.Tensor:
    """Prepare an image for EfficientNet-B4 or Xception inference."""
    pil_image = _to_pil_rgb(image)
    transform = T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = transform(pil_image)
    return _ensure_batch(tensor)


def run_univfd(model: Any, image_tensor: torch.Tensor, device: str | torch.device) -> float:
    device_obj = _normalize_device(device)
    model = _move_univfd_to_device(model, device_obj)
    model_obj = _resolve_torch_model(model)
    model_obj = _move_model_to_device(model_obj, device_obj)

    image_tensor = _ensure_batch(image_tensor).to(device_obj)

    with torch.no_grad():
        if hasattr(model, "clip_model") and hasattr(model, "linear"):
            features = model.clip_model.encode_image(image_tensor)
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
            logits = model.linear(features.float())
        else:
            logits = model_obj(image_tensor)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

    return _sigmoid_probability_from_logits(logits)


def run_efficientnet(model: Any, face_tensor: torch.Tensor, device: str | torch.device) -> float:
    device_obj = _normalize_device(device)
    model_obj = _resolve_torch_model(model)
    model_obj = _move_model_to_device(model_obj, device_obj)

    face_tensor = _ensure_batch(face_tensor).to(device_obj)
    face_tensor = F.interpolate(face_tensor, size=(380, 380), mode="bilinear", align_corners=False)

    with torch.no_grad():
        logits = model_obj(face_tensor)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

    return _sigmoid_probability_from_logits(logits)


def run_xception(model: Any, face_tensor: torch.Tensor, device: str | torch.device) -> float:
    device_obj = _normalize_device(device)
    model_obj = _resolve_torch_model(model)
    model_obj = _move_model_to_device(model_obj, device_obj)

    face_tensor = _ensure_batch(face_tensor).to(device_obj)
    face_tensor = F.interpolate(face_tensor, size=(299, 299), mode="bilinear", align_corners=False)

    with torch.no_grad():
        logits = model_obj(face_tensor)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

    return _sigmoid_probability_from_logits(logits)


def _load_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.array(image)


def debug_individual_models(image_path: str, models: dict, device: str):
    """Run the three detectors on a single image and print their scores."""
    try:
        image = _load_image(image_path)
    except Exception as exc:
        print(f"Invalid image: {image_path} ({exc})")
        return {
            "image_path": image_path,
            "face_detected": False,
            "used_full_image_for_face_models": False,
            "univfd": None,
            "efficientnet": None,
            "xception": None,
        }

    device_obj = _normalize_device(device)
    univfd_model = models.get("univfd")
    efficientnet_model = models.get("efficientnet")
    xception_model = models.get("xception")

    univfd_score = None
    efficientnet_score = None
    xception_score = None

    face = detect_largest_face(image)
    face_detected = face is not None
    used_full_image_for_face_models = not face_detected

    try:
        univfd_tensor = preprocess_univfd(image)
        if univfd_model is not None:
            univfd_score = run_univfd(univfd_model, univfd_tensor, device_obj)
    except Exception as exc:
        print(f"UnivFD failed for {image_path}: {exc}")

    face_image = face if face_detected else image

    try:
        face_tensor = preprocess_face_model(face_image)
        if efficientnet_model is not None:
            efficientnet_score = run_efficientnet(efficientnet_model, face_tensor, device_obj)
        if xception_model is not None:
            xception_score = run_xception(xception_model, face_tensor, device_obj)
    except Exception as exc:
        print(f"Face-model inference failed for {image_path}: {exc}")

    print(f"Image: {image_path}")
    print(f"Face detected: {face_detected}")
    print(f"Used full image fallback: {used_full_image_for_face_models}")
    print(f"UnivFD score: {univfd_score if univfd_score is not None else 'unavailable'}")
    print(f"EfficientNet score: {efficientnet_score if efficientnet_score is not None else 'skipped/unavailable'}")
    print(f"Xception score: {xception_score if xception_score is not None else 'skipped/unavailable'}")

    return {
        "image_path": image_path,
        "face_detected": face_detected,
        "used_full_image_for_face_models": used_full_image_for_face_models,
        "univfd": univfd_score,
        "efficientnet": efficientnet_score,
        "xception": xception_score,
    }


def validate_individual_models(
    real_dir: str,
    fake_dir: str,
    models: dict,
    device: str,
    samples_per_class: int = 5,
    seed: int = 7,
):
    """Run the debug helper on a small set of real and fake images."""
    rng = random.Random(seed)
    real_paths = _collect_image_paths(real_dir)
    fake_paths = _collect_image_paths(fake_dir)

    if not real_paths:
        print(f"No images found in real_dir: {real_dir}")
    if not fake_paths:
        print(f"No images found in fake_dir: {fake_dir}")

    selected_real = rng.sample(real_paths, k=min(samples_per_class, len(real_paths))) if real_paths else []
    selected_fake = rng.sample(fake_paths, k=min(samples_per_class, len(fake_paths))) if fake_paths else []

    results = []
    print("--- Real images ---")
    for path in selected_real:
        results.append({"label": "real", **debug_individual_models(str(path), models, device)})

    print("--- Fake images ---")
    for path in selected_fake:
        results.append({"label": "fake", **debug_individual_models(str(path), models, device)})

    return results