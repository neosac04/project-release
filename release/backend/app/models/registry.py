from __future__ import annotations

from pathlib import Path

import structlog
import torch

from app.config.settings import MODEL_PATHS
from app.models.base import BaseDetector
from app.models.efficientnet import EfficientNetDetector
from app.models.f3net import F3NetDetector
from app.models.hive_detector import HiveDetector
from app.models.siglip_detector import SigLIPDetector
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

        # parents[2] of .../backend/app/models/registry.py = .../backend
        repo_root = Path(__file__).resolve().parents[2]

        def resolve_path(path_str: str) -> Path:
            path = Path(path_str)
            return path if path.is_absolute() else (repo_root / path).resolve()

        import os
        detectors: list[tuple[str, BaseDetector, str]] = [
            ("efficientnet", EfficientNetDetector(), MODEL_PATHS["efficientnet"]),
            ("vit",          ViTDetector(),          MODEL_PATHS["vit"]),
            ("f3net",        F3NetDetector(),         MODEL_PATHS["f3net"]),
            ("siglip",       SigLIPDetector(),        MODEL_PATHS["siglip"]),
        ]
        if os.getenv("ENABLE_XCEPTIONNET") == "1":
            detectors.insert(2, ("xceptionnet", XceptionNetDetector(), MODEL_PATHS["xceptionnet"]))

        # Hive: only register if api_key is provided
        from app.config.settings import settings as _settings
        if _settings.ext_api_key:
            detectors.append(("hive", HiveDetector(), _settings.ext_api_key))

        for name, detector, weights_path in detectors:
            is_remote = weights_path.startswith("huggingface://")
            is_api_key = (name == "hive")
            resolved_path = weights_path if (is_remote or is_api_key) else str(resolve_path(weights_path))

            if is_remote or is_api_key or Path(resolved_path).exists():
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
