from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from app.explainability.gradcam import gradcam
from app.models.base import BaseDetector, ModelOutput
from app.models.calibration import calibrated_fake_prob
from app.models.xception_arch import Xception


_FAD_SIZE = 256


class _Filter(nn.Module):
    """Frequency-domain band filter with a learnable additive component (F3Net FAD head)."""

    def __init__(self, size: int = _FAD_SIZE) -> None:
        super().__init__()
        self.base = nn.Parameter(torch.zeros(size, size), requires_grad=False)
        self.learnable = nn.Parameter(torch.zeros(size, size), requires_grad=True)

    def forward(self, x_freq: torch.Tensor) -> torch.Tensor:
        # tanh on learnable, then add to base (matches F3Net paper)
        filt = self.base + torch.tanh(self.learnable)
        return x_freq * filt


class _FADHead(nn.Module):
    """
    Frequency-Aware Decomposition head.

    Input:  (B, 3, 256, 256) spatial RGB
    Output: (B, 12, 256, 256) — 4 frequency bands × 3 RGB channels
    """

    def __init__(self, size: int = _FAD_SIZE) -> None:
        super().__init__()
        self._DCT_all = nn.Parameter(torch.zeros(size, size), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.zeros(size, size), requires_grad=False)
        self.filters = nn.ModuleList([_Filter(size) for _ in range(4)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Full-image 2-D DCT: X = D @ x @ D^T
        x_freq = self._DCT_all @ x @ self._DCT_all_T
        bands: list[torch.Tensor] = []
        for filt in self.filters:
            y_freq = filt(x_freq)
            # Inverse DCT: y = D^T @ Y @ D
            y = self._DCT_all_T @ y_freq @ self._DCT_all
            bands.append(y)
        return torch.cat(bands, dim=1)


class _F3NetWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.FAD_head = _FADHead(_FAD_SIZE)
        self.backbone = Xception(num_classes=2, in_channels=12, last_linear_seq=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.FAD_head(x)
        return self.backbone(x)


class F3NetDetector(BaseDetector):
    """
    F3Net (Frequency-aware Forgery Network) — Xception backbone fed by 12-channel
    FAD-decomposed frequency bands of the full image.

    Class order: Fake=0, Real=1.
    Input: 256×256 RGB (full image, no face crop).
    """

    model_name = "f3net"

    def __init__(self) -> None:
        self._loaded = False
        self.model: _F3NetWrapper | None = None
        self.device = torch.device("cpu")

    def load(self, weights_path: str, device: torch.device) -> None:
        self.device = device
        model = _F3NetWrapper()
        state = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"⚠️ F3Net load — missing={len(missing)} unexpected={len(unexpected)}")
            if missing[:5]:
                print("   missing[:5]:", missing[:5])
            if unexpected[:5]:
                print("   unexpected[:5]:", unexpected[:5])
        else:
            print("✅ F3Net loaded cleanly (0 missing, 0 unexpected)")

        model.to(device).eval()
        self.model = model
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _prepare_tensor(self, preprocessed: dict) -> torch.Tensor:
        # F3Net wants 256×256 with DeepfakeBench normalisation (mean/std = 0.5).
        return preprocessed["dfb_tensor"].unsqueeze(0).to(self.device)

    def predict(self, preprocessed: dict) -> ModelOutput:
        assert self.model is not None
        t0 = time.time()
        with torch.no_grad():
            tensor = self._prepare_tensor(preprocessed)
            logits = self.model(tensor).squeeze(0)

        # Calibrated in logit-diff space (matches calibrate_models.py)
        logit_diff = float(logits[0] - logits[1])
        fake_prob = calibrated_fake_prob("f3net", logit_diff, domain="logit_diff")
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
