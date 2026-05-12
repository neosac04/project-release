from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_path(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


# Paths are resolved relative to the repo root by registry.py.
MODEL_PATHS = {
    "efficientnet": _env_path(
        "EFFICIENTNET_MODEL_PATH",
        "backend/app/models/weights/efficientnet_binary.pth",
    ),
    "xceptionnet": _env_path(
        "XCEPTIONNET_MODEL_PATH",
        "backend/app/models/weights/xception_best.pth",
    ),
    "f3net": _env_path(
        "F3NET_MODEL_PATH",
        "backend/app/models/weights/f3net_best.pth",
    ),
    # ViT weights are pulled from HuggingFace hub on first load. Sentinel path
    # is just checked for "non-empty"; the actual download is handled by the detector.
    "vit": "huggingface://dima806/deepfake_vs_real_image_detection",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    models_dir: str = "./models"
    device: str = "cpu"
    max_file_size_mb: int = 10

    # Verdict thresholds (final fused score)
    confidence_fake_threshold: float = 0.65
    confidence_real_threshold: float = 0.35

    # Uncertainty band — when external fallback should be consulted (future use)
    uncertainty_low: float = 0.38
    uncertainty_high: float = 0.62

    # External API (Hive Moderation) — wired but not used until configured.
    ext_api_url: str = ""
    ext_api_key: str = ""


settings = Settings()
