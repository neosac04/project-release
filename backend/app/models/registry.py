from __future__ import annotations

from pathlib import Path

import structlog
import torch

from app.config.settings import MODEL_PATHS
from app.models.base import BaseDetector
from app.models.efficientnet import EfficientNetDetector
from app.models.f3net import F3NetDetector
from app.models.vit_detector import ViTDetector
from app.models.xceptionnet import XceptionNetDetector

log = structlog.get_logger()


class ModelRegistry:
    _instance: ModelRegistry | None = None

    def __init__(self) -> None:
        self._detectors: dict[str, BaseDetector] = {}
        self._loaded = False

    @classmethod
    def get_instance(cls) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_all(self, models_dir: str, device_str: str = "cpu") -> None:
        device = torch.device(
            device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        log.info("loading_models", device=str(device))

        repo_root = Path(__file__).resolve().parents[3]

        def resolve_path(path_str: str) -> Path:
            path = Path(path_str)
            return path if path.is_absolute() else (repo_root / path).resolve()

        # Production trio: EfficientNet (locally trained head), ViT (HF
        # dima806, test AUC 0.999), F3Net (kept for heatmaps + future retrain).
        # XceptionNet is wired up but disabled — checkpoint AUC 0.475 < random.
        # Re-enable with ENABLE_XCEPTIONNET=1 for inspection only.
        import os
        detectors: list[tuple[str, BaseDetector, str]] = [
            ("efficientnet", EfficientNetDetector(), MODEL_PATHS["efficientnet"]),
            ("vit", ViTDetector(), MODEL_PATHS["vit"]),
            ("f3net", F3NetDetector(), MODEL_PATHS["f3net"]),
        ]
        if os.getenv("ENABLE_XCEPTIONNET") == "1":
            detectors.insert(2, ("xceptionnet", XceptionNetDetector(), MODEL_PATHS["xceptionnet"]))

        for name, detector, weights_path in detectors:
            # HF-hosted weights are pulled by the detector itself — skip the
            # local existence check for them.
            is_remote = weights_path.startswith("huggingface://")
            resolved_path = weights_path if is_remote else str(resolve_path(weights_path))
            if is_remote or Path(resolved_path).exists():
                try:
                    detector.load(resolved_path, device)
                    self._detectors[name] = detector
                    log.info("model_loaded", name=name)
                except Exception as exc:
                    log.warning("model_load_failed", name=name, error=str(exc))
                    print(f"❌ {name} load failed: {exc}")
            else:
                print(f"⚠️ missing weights for {name}: {resolved_path}")
                log.warning("weights_missing", name=name, path=str(resolved_path))

        self._loaded = True

    def get(self, name: str) -> BaseDetector | None:
        return self._detectors.get(name)

    def all(self) -> dict[str, BaseDetector]:
        return dict(self._detectors)

    def status(self) -> dict:
        return {
            name: {"loaded": True, "model_name": det.model_name}
            for name, det in self._detectors.items()
        }
