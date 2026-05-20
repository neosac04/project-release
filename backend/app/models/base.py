from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import torch


@dataclass
class ModelOutput:
    model_name: str
    fake_prob: float
    real_prob: float
    inference_time_ms: float
    features: np.ndarray = field(default_factory=lambda: np.array([]))
    heatmap: np.ndarray | None = None  # (H, W) float32 normalized 0-1


class BaseDetector(ABC):
    model_name: str = "base"
    device: torch.device = torch.device("cpu")

    @abstractmethod
    def load(self, weights_path: str, device: torch.device) -> None:
        pass

    @abstractmethod
    def predict(self, preprocessed: dict) -> ModelOutput:
        pass

    def get_heatmap(self, preprocessed: dict) -> np.ndarray | None:
        return None

    @property
    def is_loaded(self) -> bool:
        return False
