"""
HuggingFace ViT deepfake detector (dima806/deepfake_vs_real_image_detection).

Architecture: google/vit-base-patch16-224 fine-tuned for binary deepfake detection.
- Input: 224×224 RGB
- Class order from config.id2label: {0: 'Real', 1: 'Fake'}  → fake_prob = softmax[1]
- Test-set AUC 0.999 / 99.0% accuracy on Dataset/Test (June 2024)

Heatmaps use last-layer attention from CLS token to patch tokens (14×14 grid).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from app.explainability.gradcam import gradcam
from app.models.base import BaseDetector, ModelOutput
from app.models.calibration import calibrated_fake_prob


_HF_MODEL_ID = "dima806/deepfake_vs_real_image_detection"


class ViTDetector(BaseDetector):
    model_name = "vit"

    def __init__(self) -> None:
        self._loaded = False
        self.model = None
        self.processor = None
        self.device = torch.device("cpu")

    def load(self, weights_path: str, device: torch.device) -> None:
        # weights_path is unused — we pull from the HF hub (or local cache).
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.device = device
        print(f"🔍 Loading ViT deepfake detector ({_HF_MODEL_ID})…")
        self.processor = AutoImageProcessor.from_pretrained(_HF_MODEL_ID)
        # output_attentions=True is needed so we can extract CLS→patch attention
        # for the heatmap visualisation (otherwise the model returns an empty
        # attentions tuple and the heatmap endpoint returns 503).
        self.model = AutoModelForImageClassification.from_pretrained(
            _HF_MODEL_ID,
            output_attentions=True,
        )
        self.model.to(device).eval()

        id2label = self.model.config.id2label
        fake_indices = [int(i) for i, lbl in id2label.items() if "fake" in lbl.lower()]
        if not fake_indices:
            raise RuntimeError(f"Could not locate 'fake' class in id2label: {id2label}")
        self._fake_idx = fake_indices[0]
        self._real_idx = 1 - self._fake_idx
        print(f"✅ ViT loaded (fake_idx={self._fake_idx}, label='{id2label[self._fake_idx]}')")

        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _preprocess_pil(self, preprocessed: dict) -> dict:
        pil = preprocessed["pil"]
        inputs = self.processor(images=pil, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def predict(self, preprocessed: dict) -> ModelOutput:
        assert self.model is not None
        t0 = time.time()
        with torch.no_grad():
            inputs = self._preprocess_pil(preprocessed)
            logits = self.model(**inputs).logits.squeeze(0)
            probs = torch.softmax(logits, dim=-1)

        raw_fake = float(probs[self._fake_idx])
        # Apply calibration if available, otherwise pass through
        fake_prob = calibrated_fake_prob("vit", raw_fake, domain="fake_prob")
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
        """Attention-based saliency: CLS-token attention to image patches, last layer."""
        if self.model is None:
            return None
        try:
            inputs = self._preprocess_pil(preprocessed)
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
            if not outputs.attentions:
                print("⚠️ ViT heatmap: model returned empty attentions tuple")
                return None
            attn = outputs.attentions[-1]  # (1, heads, seq, seq)
            cls_to_patches = attn[0, :, 0, 1:].mean(dim=0)  # avg over heads → (196,)
            grid = cls_to_patches.reshape(14, 14).cpu().numpy()
            mn, mx = float(grid.min()), float(grid.max())
            return ((grid - mn) / (mx - mn + 1e-8)).astype(np.float32)
        except Exception as exc:
            print(f"⚠️ ViT heatmap failed: {exc!r}")
            return None
