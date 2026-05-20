"""Loads per-model Platt-scaling parameters from calibration.json."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

_CALIB_CACHE: dict[str, dict] | None = None


def _calib_path() -> Path:
    return Path(__file__).resolve().parent / "weights" / "calibration.json"


def _load() -> dict[str, dict]:
    global _CALIB_CACHE
    if _CALIB_CACHE is not None:
        return _CALIB_CACHE
    path = _calib_path()
    if not path.exists():
        _CALIB_CACHE = {}
    else:
        _CALIB_CACHE = json.loads(path.read_text())
    return _CALIB_CACHE


def calibrated_fake_prob(
    model_name: str,
    raw: float,
    domain: Literal["fake_prob", "logit_diff"],
) -> float:
    """
    Apply sigmoid(a * raw + b). Returns raw if no calibration exists for this model.

    `domain` must match the calibration file (raises on mismatch).
    """
    calib = _load().get(model_name)
    if not calib or "a" not in calib:
        # No calibration recorded — pass through
        if domain == "fake_prob":
            return raw
        return 1.0 / (1.0 + math.exp(-raw))

    if calib.get("domain") != domain:
        raise ValueError(
            f"Calibration domain mismatch for {model_name}: "
            f"file={calib.get('domain')} vs runtime={domain}"
        )

    z = calib["a"] * raw + calib["b"]
    # Clip to avoid overflow
    z = max(min(z, 30.0), -30.0)
    return float(1.0 / (1.0 + math.exp(-z)))
