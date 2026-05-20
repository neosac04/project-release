"""
Hive AI deepfake detection model.

Calls the Hive AI deepfake-image-detection API and maps the result to a
fake_prob score. Only registered in the ensemble if `ext_api_key` is set.

API docs: https://docs.thehive.ai/reference/deepfake-image-detection
Endpoint: POST https://api.thehive.ai/api/v2/task/sync
Header  : Authorization: Token <key>
Body    : multipart/form-data — field "media"
"""
from __future__ import annotations

import io
import time

import numpy as np

from app.models.base import BaseDetector, ModelOutput

_HIVE_ENDPOINT = "https://api.thehive.ai/api/v2/task/sync"

_FAKE_CLASSES = {
    "yes_ai_generated",
    "ai_generated",
    "deepfake",
    "yes_deepfake",
}


class HiveDetector(BaseDetector):
    model_name = "hive"

    def __init__(self) -> None:
        self._loaded = False
        self._api_key: str = ""

    def load(self, weights_path: str, device) -> None:
        self._api_key = weights_path  # registry passes the api_key here
        if not self._api_key:
            raise ValueError("Hive API key is empty — set ext_api_key in settings.")
        print(f"✅ Hive detector ready (key: …{self._api_key[-4:]})")
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, preprocessed: dict) -> ModelOutput:
        import requests

        t0 = time.time()
        pil = preprocessed["pil"]

        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=92)
        buf.seek(0)

        fake_prob = 0.5
        try:
            resp = requests.post(
                _HIVE_ENDPOINT,
                headers={"Authorization": f"Token {self._api_key}"},
                files={"media": ("image.jpg", buf, "image/jpeg")},
                timeout=20,
            )
            resp.raise_for_status()
            fake_prob = _parse_hive_response(resp.json())
        except Exception as exc:
            print(f"⚠️ Hive API call failed: {exc!r} — using neutral score 0.5")

        fake_prob = max(min(float(fake_prob), 0.999), 0.001)
        elapsed = (time.time() - t0) * 1000

        return ModelOutput(
            model_name=self.model_name,
            fake_prob=fake_prob,
            real_prob=1.0 - fake_prob,
            inference_time_ms=elapsed,
            features=np.array([fake_prob]),
        )


def _parse_hive_response(data: dict) -> float:
    try:
        classes: list[dict] = data["output"][0]["classes"]
        score_map = {c["class"]: float(c["score"]) for c in classes}
        fake_total = sum(v for k, v in score_map.items() if k in _FAKE_CLASSES)
        if fake_total > 0:
            return fake_total
        real_keys = [k for k in score_map if k not in _FAKE_CLASSES]
        if real_keys:
            return 1.0 - min(sum(score_map[k] for k in real_keys), 1.0)
    except (KeyError, IndexError, TypeError):
        pass
    return 0.5
