from __future__ import annotations
import time
import torch
import torch.nn as nn
import torchvision.models as tvm
from app.models.base import BaseDetector, ModelOutput


class DIREDetector(BaseDetector):
    """
    DistilDIRE: ResNet-50 classifier trained on diffusion reconstruction errors.
    Catches diffusion-generated and GAN images that fool CNN classifiers.
    """

    model_name = "dire"

    def __init__(self) -> None:
        self._loaded = False
        self.model: nn.Module | None = None
        self.device = torch.device("cpu")

    def load(self, weights_path: str, device: torch.device) -> None:
        self.device = device
        self.model = tvm.resnet50(weights=None)
        self.model.fc = nn.Linear(2048, 2)
        state = torch.load(weights_path, map_location=device)
        if isinstance(state, dict):
            sd = state.get("model", state.get("state_dict", state))
        else:
            sd = state
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=False)
        self.model.eval().to(device)
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, preprocessed: dict) -> ModelOutput:
        t0 = time.time()
        # DistilDIRE uses ImageNet-normalized input
        tensor = preprocessed["imagenet_tensor"].unsqueeze(0).to(self.device)
        # Resize to 224 (ResNet standard)
        import torch.nn.functional as F
        tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze()
        elapsed = (time.time() - t0) * 1000
        return ModelOutput(
            model_name=self.model_name,
            fake_prob=probs[1].item(),
            real_prob=probs[0].item(),
            inference_time_ms=elapsed,
            features=logits.cpu().numpy().flatten(),
        )
