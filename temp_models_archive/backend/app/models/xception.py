from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from app.models.base import BaseDetector, ModelOutput
from app.utils.model_loading import load_model_weights


class XceptionDetector(BaseDetector):
    """
    Xception trained on FaceForensics++ (DeepfakeBench checkpoint).
    Xception is the standard baseline from the original FaceForensics++ paper.
    Uses separable depthwise convolutions — excellent at detecting texture artifacts.
    GradCAM++ target: final exit_flow conv layer.
    """

    model_name = "xception"

    def __init__(self) -> None:
        self._loaded = False
        self.model: nn.Module | None = None
        self.device = torch.device("cpu")
        self._gradcam_layer: nn.Module | None = None
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

    def load(self, weights_path: str, device: torch.device) -> None:
        import timm
        self.device = device
        # timm's xception has num_classes as the classification head size
        self.model = timm.create_model("xception", pretrained=False, num_classes=2)
        self.model = load_model_weights(self.model, weights_path, str(device))

        # GradCAM++ target: last conv layer before global pool
        try:
            self._gradcam_layer = self.model.act4  # timm xception final act layer
        except AttributeError:
            try:
                self._gradcam_layer = list(self.model.children())[-3]
            except Exception:
                self._gradcam_layer = None

        if self._gradcam_layer is not None:
            self._register_hooks()
        self._loaded = True

    def _register_hooks(self) -> None:
        def fwd_hook(module, inp, out):
            self._activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._gradcam_layer.register_forward_hook(fwd_hook)
        self._gradcam_layer.register_full_backward_hook(bwd_hook)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, preprocessed: dict) -> ModelOutput:
        t0 = time.time()
        with torch.no_grad():
            # Xception native input: 299×299
            tensor = preprocessed["imagenet_tensor"].unsqueeze(0).to(self.device)
            tensor = F.interpolate(tensor, size=(299, 299), mode="bilinear", align_corners=False)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze()
        elapsed = (time.time() - t0) * 1000
        return ModelOutput(
            model_name=self.model_name,
            fake_prob=float(probs[1]),
            real_prob=float(probs[0]),
            inference_time_ms=elapsed,
            features=logits.cpu().numpy().flatten(),
        )

    def get_heatmap(self, preprocessed: dict) -> np.ndarray | None:
        if self._gradcam_layer is None:
            return None
        try:
            tensor = preprocessed["imagenet_tensor"].unsqueeze(0).to(self.device)
            tensor = F.interpolate(tensor, size=(299, 299), mode="bilinear", align_corners=False)
            tensor = tensor.detach().requires_grad_(True)

            logits = self.model(tensor)
            self.model.zero_grad()
            one_hot = torch.zeros_like(logits)
            one_hot[0, 1] = 1.0
            logits.backward(gradient=one_hot, retain_graph=False)

            acts = self._activations
            grads = self._gradients
            if acts is None or grads is None:
                return None

            # Standard GradCAM (simpler, more stable for Xception)
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * acts).sum(dim=1)).squeeze()
            cam = cam.detach().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam.astype(np.float32)
        except Exception:
            return None
