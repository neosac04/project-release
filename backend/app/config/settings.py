from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_path(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


# Paths resolved relative to the backend/ root by registry.py.
MODEL_PATHS = {
    "efficientnet": _env_path(
        "EFFICIENTNET_MODEL_PATH",
        "app/models/weights/efficientnet_binary.pth",
    ),
    "xceptionnet": _env_path(
        "XCEPTIONNET_MODEL_PATH",
        "app/models/weights/xception_best.pth",
    ),
    "f3net": _env_path(
        "F3NET_MODEL_PATH",
        "app/models/weights/f3net_binary_best.pth",
    ),
    # ViT pulled from HuggingFace hub on first load.
    "vit": "huggingface://dima806/deepfake_vs_real_image_detection",
    # SigLIP — locally fine-tuned binary classifier (94.44% acc).
    "siglip": _env_path(
        "SIGLIP_MODEL_PATH",
        "app/models/weights/siglip",
    ),
    # Hive API — key injected at runtime via ext_api_key setting.
    "hive": "hive_api_key_placeholder",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    models_dir: str = "./models"
    device: str = "cpu"
    max_file_size_mb: int = 10

    # Verdict thresholds
    confidence_fake_threshold: float = 0.65
    confidence_real_threshold: float = 0.35

    # Uncertainty band
    uncertainty_low: float = 0.38
    uncertainty_high: float = 0.62

    # Video-specific settings
    video_frames_to_sample: int = 32
    max_video_size_mb: int = 100

    # External API (Hive) — activates Hive model when ext_api_key is set.
    ext_api_url: str = ""
    ext_api_key: str = ""


settings = Settings()
