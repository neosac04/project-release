from __future__ import annotations

import importlib
import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.models.registry import ModelRegistry

router = APIRouter()

def _find_mediapipe_asset() -> Path:
    """
    Locate the MediaPipe face_landmarker.task file.

    Resolution order:
      1. MEDIAPIPE_MODEL_PATH environment variable (explicit override)
      2. Walk up from this file until we find a 'models/' sibling directory
         (handles any depth inside backend/)
      3. Fall back to a path that will simply not exist (``available=False``)
    """
    env_override = os.environ.get("MEDIAPIPE_MODEL_PATH")
    if env_override:
        return Path(env_override)

    # Walk upward from this file's directory until we find a sibling 'models/'
    current = Path(__file__).resolve().parent
    for _ in range(10):   # safety limit — won't climb past filesystem root
        candidate = current / "models" / "mediapipe" / "face_landmarker.task"
        if candidate.exists():
            return candidate
        models_dir = current / "models"
        if models_dir.is_dir():
            # Found the models dir — even if the .task file is not yet there
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent

    # Final fallback — will be reported as missing in /ready
    return Path(__file__).resolve().parents[3] / "models" / "mediapipe" / "face_landmarker.task"


_MEDIAPIPE_ASSET = _find_mediapipe_asset()


# ---------------------------------------------------------------------------
# /health  — liveness probe (always fast, always 200 when process is alive)
# ---------------------------------------------------------------------------

@router.get("/health", summary="Liveness probe")
async def health():
    """
    Lightweight liveness check.  Returns 200 as long as the process is alive.
    Use ``/ready`` to know whether the full inference pipeline is operational.
    """
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# /ready  — readiness probe (reports model + asset status)
# ---------------------------------------------------------------------------

@router.get("/ready", summary="Readiness probe")
async def ready():
    """
    Reports whether the backend is ready to serve inference requests.

    Checks:
    - Which ML models are loaded in the registry
    - Whether the MediaPipe FaceLandmarker asset exists on disk
    - Whether PyTorch is importable

    Returns HTTP 200 when fully ready, HTTP 503 when not.
    """
    registry = ModelRegistry.get_instance()
    loaded_models = list(registry.all().keys())

    expected_models = ["efficientnet", "vit", "f3net"]
    missing_models = [m for m in expected_models if m not in loaded_models]

    mediapipe_asset_ok = _MEDIAPIPE_ASSET.exists()

    torch_ok = _check_import("torch")
    mediapipe_ok = _check_import("mediapipe")

    issues: list[str] = []
    if missing_models:
        issues.append(f"Models not loaded: {missing_models}.")
    if not mediapipe_asset_ok:
        issues.append(
            f"MediaPipe asset missing at {_MEDIAPIPE_ASSET}. "
            "Face detection will fall back to Haar Cascade."
        )
    if not torch_ok:
        issues.append("PyTorch not importable. Check your Python environment.")
    if not mediapipe_ok:
        issues.append("mediapipe package not installed. Run: pip install mediapipe==0.10.14")

    is_ready = len(issues) == 0 or (
        # Degraded-but-functional: at least one model loaded, torch works
        len(loaded_models) > 0 and torch_ok
    )

    payload = {
        "ready": is_ready,
        "models_loaded": loaded_models,
        "models_missing": missing_models,
        "mediapipe_asset_present": mediapipe_asset_ok,
        "torch_available": torch_ok,
        "mediapipe_package_available": mediapipe_ok,
        "issues": issues,
    }

    status_code = 200 if is_ready else 503
    return JSONResponse(content=payload, status_code=status_code)


# ---------------------------------------------------------------------------
# /models/status  — existing endpoint, kept for compatibility
# ---------------------------------------------------------------------------

@router.get("/models/status", summary="Detailed model registry status")
async def models_status():
    registry = ModelRegistry.get_instance()
    return {"models": registry.status()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_import(package: str) -> bool:
    try:
        importlib.import_module(package)
        return True
    except ImportError:
        return False


