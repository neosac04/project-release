from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from app.explainability.gradcam import gradcam
from app.models.base import BaseDetector, ModelOutput
from app.models.xception_arch import Xception


class _XceptionWrapper(nn.Module):
    """Wraps Xception under `backbone.*` to match the DeepfakeBench checkpoint layout."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = Xception(num_classes=2, in_channels=3, last_linear_seq=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class XceptionNetDetector(BaseDetector):
    """
    DeepfakeBench-trained XceptionNet for binary deepfake detection.

    Class order: Fake=0, Real=1.
    Input: 299×299 RGB, ImageNet normalisation (face crop).
    """

    model_name = "xceptionnet"

    def __init__(self) -> None:
        self._loaded = False
        self.model: _XceptionWrapper | None = None
        self.device = torch.device("cpu")

    def load(self, weights_path: str, device: torch.device) -> None:
        self.device = device
        model = _XceptionWrapper()
        state = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"⚠️ XceptionNet load — missing={len(missing)} unexpected={len(unexpected)}"
            )
            if missing[:5]:
                print("   missing[:5]:", missing[:5])
            if unexpected[:5]:
                print("   unexpected[:5]:", unexpected[:5])
        else:
            print("✅ XceptionNet loaded cleanly (0 missing, 0 unexpected)")

        model.to(device).eval()
        self.model = model
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _prepare_tensor(self, preprocessed: dict) -> torch.Tensor:
        # DeepfakeBench Xception was trained at 256×256 with mean/std = 0.5.
        return preprocessed["dfb_tensor"].unsqueeze(0).to(self.device)

    def predict(self, preprocessed: dict) -> ModelOutput:
        assert self.model is not None
        t0 = time.time()
        with torch.no_grad():
            tensor = self._prepare_tensor(preprocessed)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        fake_prob = float(probs[0])
        real_prob = float(probs[1])
        fake_prob = max(min(fake_prob, 0.999), 0.001)
        real_prob = 1.0 - fake_prob

        elapsed = (time.time() - t0) * 1000
        return ModelOutput(
            model_name=self.model_name,
            fake_prob=fake_prob,
            real_prob=real_prob,
            inference_time_ms=elapsed,
            features=logits.detach().cpu().numpy().flatten(),
        )

    def get_heatmap(self, preprocessed: dict) -> np.ndarray | None:
        if self.model is None:
            return None
        tensor = self._prepare_tensor(preprocessed)
        return gradcam(self.model, "backbone.bn4", tensor, self.device, target_class=0)
