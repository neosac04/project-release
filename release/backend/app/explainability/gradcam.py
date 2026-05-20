from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _resolve_layer(model: nn.Module, dotted_name: str) -> nn.Module:
    module: nn.Module = model
    for part in dotted_name.split("."):
        if not hasattr(module, part):
            raise RuntimeError(f"GradCAM target layer not found: {dotted_name}")
        module = getattr(module, part)
    return module


def gradcam(
    model: nn.Module,
    layer_name: str,
    tensor: torch.Tensor,
    device: torch.device,
    target_class: int = 0,
) -> np.ndarray | None:
    """
    Standard GradCAM.

    - `tensor`: pre-batched input (B, C, H, W) — caller is responsible for shape.
    - `target_class`: class index to back-propagate from. Default 0 == Fake.
    """
    layer = _resolve_layer(model, layer_name)

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def fwd_hook(_module, _inp, out):
        activations["a"] = out

    def bwd_hook(_module, _grad_in, grad_out):
        gradients["g"] = grad_out[0].detach()

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    try:
        model.eval()
        tensor = tensor.to(device)
        tensor = tensor.detach().requires_grad_(True)

        logits = model(tensor)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0
        model.zero_grad()
        logits.backward(gradient=one_hot, retain_graph=False)

        acts = activations.get("a")
        grads = gradients.get("g")
        if acts is None or grads is None:
            return None

        acts = acts.detach()
        # GradCAM: weight each channel by global-avg of gradients.
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1).squeeze(0)
        cam = torch.relu(cam).cpu().numpy()

        cam_min = float(cam.min())
        cam_max = float(cam.max())
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.astype(np.float32)
    except Exception:
        return None
    finally:
        h1.remove()
        h2.remove()
