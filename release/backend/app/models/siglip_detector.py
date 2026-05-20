"""
SigLIP deepfake detector.

Loads a locally fine-tuned SiglipForImageClassification model.
- Architecture : google/siglip-base-patch16-224 (fine-tuned binary classifier)
- Classes      : id2label = {0: "Fake", 1: "Real"}  →  fake_idx = 0
- Input        : 224×224 RGB, mean/std = [0.5, 0.5, 0.5]
- Heatmap      : gradient × input saliency (backprop through softmax)
- Accuracy     : 94.44 % on 19 999-image held-out test set
"""
from __future__ import annotations

import time

import numpy as np
import torch

from app.models.base import BaseDetector, ModelOutput
from app.models.calibration import calibrated_fake_prob


class SigLIPDetector(BaseDetector):
    model_name = "siglip"

    def __init__(self) -> None:
        self._loaded = False
        self.model = None
        self.processor = None
        self.device = torch.device("cpu")
        self._fake_idx: int = 0
        self._real_idx: int = 1

    def load(self, weights_path: str, device: torch.device) -> None:
        from transformers import AutoImageProcessor, SiglipForImageClassification

        self.device = device
        print(f"🔍 Loading SigLIP detector from {weights_path} …")

        self.processor = AutoImageProcessor.from_pretrained(weights_path)
        self.model = SiglipForImageClassification.from_pretrained(weights_path)
        self.model.to(device).eval()

        id2label = self.model.config.id2label
        fake_indices = [int(i) for i, lbl in id2label.items() if "fake" in lbl.lower()]
        if fake_indices:
            self._fake_idx = fake_indices[0]
            self._real_idx = 1 - self._fake_idx
        else:
            self._fake_idx, self._real_idx = 0, 1

        print(f"✅ SigLIP loaded (fake_idx={self._fake_idx}, label='{id2label[self._fake_idx]}')")
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _to_inputs(self, preprocessed: dict) -> dict:
        pil = preprocessed["pil"]
        inputs = self.processor(images=pil, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def predict(self, preprocessed: dict) -> ModelOutput:
        assert self.model is not None
        t0 = time.time()

        with torch.no_grad():
            inputs = self._to_inputs(preprocessed)
            logits = self.model(**inputs).logits.squeeze(0)
            probs = torch.softmax(logits, dim=-1)

        raw_fake = float(probs[self._fake_idx])
        fake_prob = calibrated_fake_prob("siglip", raw_fake, domain="fake_prob")
        fake_prob = max(min(fake_prob, 0.999), 0.001)

        elapsed = (time.time() - t0) * 1000
        return ModelOutput(
            model_name=self.model_name,
            fake_prob=fake_prob,
            real_prob=1.0 - fake_prob,
            inference_time_ms=elapsed,
            features=logits.detach().cpu().numpy().flatten(),
        )

    def get_heatmap(self, preprocessed: dict) -> np.ndarray | None:
        """Gradient × input saliency map."""
        if self.model is None:
            return None
        try:
            pil = preprocessed["pil"]
            inputs = self.processor(images=pil, return_tensors="pt")
            px = inputs["pixel_values"].to(self.device).requires_grad_(True)

            self.model.zero_grad()
            logits = self.model(pixel_values=px).logits
            probs = torch.softmax(logits, dim=1)
            probs[0, self._fake_idx].backward()

            if px.grad is None:
                return None

            sal = (px.grad[0].abs() * px[0].detach().abs()).mean(dim=0).cpu().numpy()
            mn, mx = float(sal.min()), float(sal.max())
            return ((sal - mn) / (mx - mn + 1e-8)).astype(np.float32)
        except Exception as exc:
            print(f"⚠️ SigLIP heatmap failed: {exc!r}")
            return None
