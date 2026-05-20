from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def _unwrap_checkpoint(checkpoint: Any) -> Any:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model", "net", "weights"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                return candidate
    return checkpoint


def _normalize_state_dict(state_dict: Any) -> dict[str, Any]:
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint does not contain a state_dict mapping")

    cleaned: dict[str, Any] = {}
    for raw_key, value in state_dict.items():
        key = str(raw_key)
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("backbone.efficientnet."):
            key = key[len("backbone.efficientnet."):]
        if key.startswith("backbone."):
            key = key[len("backbone."):]
        if key.startswith("model."):
            key = key[len("model."):]
        if key.startswith("net."):
            key = key[len("net."):]
        key = key.replace("last_layer.", "_fc.")
        key = key.replace("last_linear.", "fc.")
        cleaned[key] = value

    return cleaned


def load_model_weights(model: Any, weight_path: str, device: str):
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = _unwrap_checkpoint(checkpoint)
    cleaned_state_dict = _normalize_state_dict(state_dict)

    if isinstance(model, torch.nn.Linear):
        linear_cleaned: dict[str, Any] = {}
        for key, value in cleaned_state_dict.items():
            linear_key = key[3:] if key.startswith("fc.") else key
            linear_cleaned[linear_key] = value
        cleaned_state_dict = linear_cleaned

    if not hasattr(model, "load_state_dict"):
        raise TypeError(f"Model of type {type(model)!r} does not support load_state_dict")

    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"Loaded weights from {weight_path}")
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    return model