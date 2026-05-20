from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from app.models.base import BaseDetector, ModelOutput


class _ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x))


class _FreqNetArch(nn.Module):
    """Lightweight frequency-domain deepfake detector (1.9M params)."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer1 = nn.Sequential(_ResBlock(32), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(_ResBlock(64) if False else nn.Sequential(
            nn.Conv2d(32, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), _ResBlock(64)
        ), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            _ResBlock(128), nn.MaxPool2d(2)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.classifier(x)


class FreqNetDetector(BaseDetector):
    """
    FreqNet: AAAI 2024 — operates on DCT-transformed input.
    Forces the model to learn frequency-domain forgery artifacts.
    """

    model_name = "freqnet"

    def __init__(self) -> None:
        self._loaded = False
        self.model: _FreqNetArch | None = None
        self.device = torch.device("cpu")

    def load(self, weights_path: str, device: torch.device) -> None:
        self.device = device
        self.model = _FreqNetArch().to(device)
        state = torch.load(weights_path, map_location=device)
        if isinstance(state, dict):
            sd = state.get("model", state.get("state_dict", state))
        else:
            sd = state
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=False)
        self.model.eval()
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _dct2(self, arr: np.ndarray) -> np.ndarray:
        from scipy.fftpack import dct
        return dct(dct(arr, axis=0, norm="ortho"), axis=1, norm="ortho")

    def _to_freq_tensor(self, pil_img) -> torch.Tensor:
        import numpy as np
        img = np.array(pil_img).astype(np.float64) / 255.0  # (256,256,3)
        channels = []
        for c in range(3):
            d = self._dct2(img[:, :, c])
            d = np.log(np.abs(d) + 1e-8)
            channels.append(d)
        arr = np.stack(channels, axis=0).astype(np.float32)
        # Normalize each channel
        for c in range(3):
            mn, mx = arr[c].min(), arr[c].max()
            arr[c] = (arr[c] - mn) / (mx - mn + 1e-8)
        return torch.from_numpy(arr)

    def predict(self, preprocessed: dict) -> ModelOutput:
        t0 = time.time()
        freq_tensor = self._to_freq_tensor(preprocessed["freq_pil"])
        freq_tensor = freq_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(freq_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze()
        elapsed = (time.time() - t0) * 1000
        return ModelOutput(
            model_name=self.model_name,
            fake_prob=probs[1].item(),
            real_prob=probs[0].item(),
            inference_time_ms=elapsed,
            features=logits.cpu().numpy().flatten(),
        )

    def get_heatmap(self, preprocessed: dict) -> np.ndarray | None:
        """GradCAM on last layer of FreqNet for fake class."""
        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def fwd(m, i, o):
            activations.append(o)

        def bwd(m, gi, go):
            gradients.append(go[0])

        target_layer = self.model.layer3[0]
        h1 = target_layer.register_forward_hook(fwd)
        h2 = target_layer.register_full_backward_hook(bwd)

        try:
            freq_tensor = self._to_freq_tensor(preprocessed["freq_pil"])
            freq_tensor = freq_tensor.unsqueeze(0).to(self.device)
            freq_tensor.requires_grad_(True)
            logits = self.model(freq_tensor)
            self.model.zero_grad()
            one_hot = torch.zeros_like(logits)
            one_hot[0, 1] = 1.0
            logits.backward(gradient=one_hot)

            acts = activations[0]
            grads = gradients[0]
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1).squeeze()
            cam = F.relu(cam).detach().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam.astype(np.float32)
        except Exception:
            return None
        finally:
            h1.remove()
            h2.remove()
