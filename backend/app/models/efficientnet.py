from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from app.models.base import BaseDetector, ModelOutput
from app.models.calibration import calibrated_fake_prob

DEBUG_EFFNET_LOADING = False


class EfficientNetDetector(BaseDetector):
    """
    EfficientNet-B4 trained on FaceForensics++ (DeepfakeBench v1.0.1).
    Handles flexible checkpoint formats and provides GradCAM++ heatmaps.
    """

    model_name = "efficientnet"

    def __init__(self) -> None:
        self._loaded = False
        self.model: nn.Module | None = None
        self.device = torch.device("cpu")
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._architecture = "efficientnet_pytorch"

    def load(self, weights_path: str, device: torch.device) -> None:
        self.device = device
        print(f"\n🔍 Loading EfficientNet weights from: {weights_path}")

        # 🔹 Load checkpoint
        state = torch.load(weights_path, map_location=device)
        if isinstance(state, dict):
            sd = state.get("model", state.get("state_dict", state))
        else:
            sd = state

        if DEBUG_EFFNET_LOADING:
            print("sd.keys()")
            print(sd.keys())

        # Try efficientnet_pytorch mapping first.
        self.model, self._architecture = self._build_efficientnet_pytorch(device)
        cleaned_sd = {}
        classifier_map_hits: list[tuple[str, str]] = []

        for k, v in sd.items():
            nk = k
            for prefix in ("module.", "backbone.", "model.", "net.", "encoder.", "efficientnet."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]

            # Classifier remaps for different wrappers.
            if nk.startswith("last_layer."):
                mapped = nk.replace("last_layer.", "_fc.")
                classifier_map_hits.append((k, mapped))
                nk = mapped
            elif nk.startswith("classifier.") and not nk.startswith("classifier.0"):
                mapped = nk.replace("classifier.1.", "_fc.")
                if mapped == nk:
                    mapped = nk.replace("classifier.", "_fc.")
                classifier_map_hits.append((k, mapped))
                nk = mapped
            elif nk.startswith("head.fc."):
                mapped = nk.replace("head.fc.", "_fc.")
                classifier_map_hits.append((k, mapped))
                nk = mapped
            elif nk.startswith("fc."):
                mapped = nk.replace("fc.", "_fc.")
                classifier_map_hits.append((k, mapped))
                nk = mapped

            cleaned_sd[nk] = v

        missing, unexpected, model_keys, ckpt_keys = self._load_with_debug(cleaned_sd)

        # If this is timm/torchvision style (features.* + classifier.*), load with torchvision backend.
        timm_like = any(k.startswith("features.") for k in sd.keys()) and any(
            k.startswith("classifier.") for k in sd.keys()
        )
        if timm_like and (len(missing) > 100 or len(unexpected) > 100):
            self.model, self._architecture = self._build_torchvision_b4(device)
            cleaned_sd = self._remap_torchvision_style(sd)
            missing, unexpected, model_keys, ckpt_keys = self._load_with_debug(cleaned_sd)

        if DEBUG_EFFNET_LOADING:
            print("Classifier mappings:")
            for src, dst in classifier_map_hits[:20]:
                print(f"{src} -> {dst}")
            print("Missing from checkpoint:")
            print(model_keys - ckpt_keys)
            print("Unexpected in checkpoint:")
            print(ckpt_keys - model_keys)
            print("Tensor shapes:")
            for k, v in cleaned_sd.items():
                print(k, v.shape)

        print("✅ EfficientNet loaded")
        print("Missing keys:", missing[:10])
        print("Unexpected keys:", unexpected[:10])

        if len(missing) > 50:
            print("⚠️ WARNING: Too many missing keys — possible architecture mismatch")

        self.model.eval()

        # 🔹 GradCAM++ setup
        self._gradcam_layer = self._resolve_gradcam_layer()
        self._register_hooks()

        self._loaded = True

    def _build_efficientnet_pytorch(self, device: torch.device) -> tuple[nn.Module, str]:
        from efficientnet_pytorch import EfficientNet

        model = EfficientNet.from_name("efficientnet-b4")
        model._fc = nn.Linear(model._fc.in_features, 2)
        return model.to(device), "efficientnet_pytorch"

    def _build_torchvision_b4(self, device: torch.device) -> tuple[nn.Module, str]:
        from torchvision.models import efficientnet_b4

        model = efficientnet_b4(weights=None)
        in_feats = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feats, 2)
        return model.to(device), "torchvision"

    def _remap_torchvision_style(self, sd: dict) -> dict:
        cleaned: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            nk = k
            for prefix in ("module.", "backbone.", "model.", "net.", "encoder.", "efficientnet."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            if nk.startswith("last_layer."):
                nk = nk.replace("last_layer.", "classifier.1.")
            elif nk.startswith("_fc."):
                nk = nk.replace("_fc.", "classifier.1.")
            elif nk.startswith("head.fc."):
                nk = nk.replace("head.fc.", "classifier.1.")
            elif nk.startswith("fc."):
                nk = nk.replace("fc.", "classifier.1.")
            cleaned[nk] = v

        # If checkpoint has 1000-class classifier, intentionally skip loading it into 2-class head.
        cls_w = cleaned.get("classifier.1.weight")
        cls_b = cleaned.get("classifier.1.bias")
        if cls_w is not None and cls_w.shape[0] != 2:
            cleaned.pop("classifier.1.weight", None)
        if cls_b is not None and cls_b.shape[0] != 2:
            cleaned.pop("classifier.1.bias", None)
        return cleaned

    def _load_with_debug(self, cleaned_sd: dict) -> tuple[list[str], list[str], set[str], set[str]]:
        model_state = self.model.state_dict()
        filtered_sd: dict[str, torch.Tensor] = {}
        for k, v in cleaned_sd.items():
            if k not in model_state:
                continue
            if model_state[k].shape != v.shape:
                if DEBUG_EFFNET_LOADING:
                    print(f"Skipping shape-mismatch key: {k} ckpt={tuple(v.shape)} model={tuple(model_state[k].shape)}")
                continue
            filtered_sd[k] = v

        model_keys = set(model_state.keys())
        ckpt_keys = set(filtered_sd.keys())
        missing, unexpected = self.model.load_state_dict(filtered_sd, strict=False)
        return missing, unexpected, model_keys, ckpt_keys

    def _resolve_gradcam_layer(self) -> nn.Module:
        if hasattr(self.model, "_conv_head"):
            return self.model._conv_head
        if hasattr(self.model, "features"):
            return self.model.features[-1]
        raise RuntimeError("Unable to resolve EfficientNet GradCAM layer.")

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
            tensor = preprocessed["imagenet_tensor"].unsqueeze(0).to(self.device)
            tensor = F.interpolate(tensor, size=(380, 380), mode="bilinear", align_corners=False)

            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze()

        # Class order from training (ImageFolder on Dataset/): Fake=0, Real=1
        raw_fake = float(probs[0])
        # Apply Platt-scaling calibration if available
        fake_prob = calibrated_fake_prob("efficientnet", raw_fake, domain="fake_prob")
        fake_prob = max(min(fake_prob, 0.999), 0.001)
        real_prob = 1.0 - fake_prob

        elapsed = (time.time() - t0) * 1000

        return ModelOutput(
            model_name=self.model_name,
            fake_prob=fake_prob,
            real_prob=real_prob,
            inference_time_ms=elapsed,
            features=logits.cpu().numpy().flatten(),
        )

    def get_heatmap(self, preprocessed: dict) -> np.ndarray | None:
        """GradCAM++ for fake class (index 1)."""
        try:
            tensor = preprocessed["imagenet_tensor"].unsqueeze(0).to(self.device)
            tensor = F.interpolate(tensor, size=(380, 380), mode="bilinear", align_corners=False)
            tensor = tensor.detach().requires_grad_(True)

            logits = self.model(tensor)
            self.model.zero_grad()

            one_hot = torch.zeros_like(logits)
            one_hot[0, 0] = 1.0  # Fake class
            logits.backward(gradient=one_hot, retain_graph=False)

            acts = self._activations
            grads = self._gradients

            if acts is None or grads is None:
                return None

            # GradCAM++
            grad_sq = grads ** 2
            grad_cu = grads ** 3
            sum_acts = acts.sum(dim=(2, 3), keepdim=True)

            denom = 2 * grad_sq + sum_acts * grad_cu
            denom = torch.where(denom != 0, denom, torch.ones_like(denom))

            alpha = grad_sq / denom
            weights = (alpha * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)

            cam = (weights * acts).sum(dim=1).squeeze()
            cam = torch.relu(cam).detach().cpu().numpy()

            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            return cam.astype(np.float32)

        except Exception:
            return None