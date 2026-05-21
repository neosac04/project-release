"""
Hive AI deepfake detection via the V3 Vision-Language Model (VLM).

The V3 Playground key uses the OpenAI-compatible chat completions API
with model `hive/vision-language-model`.  The image is sent base64-encoded
inside the user message and the model is prompted to return a 0-1 fake
probability as a bare decimal.

Endpoint : POST https://api.thehive.ai/api/v3/chat/completions
Auth     : Authorization: Bearer <secret_key>
Model    : hive/vision-language-model

Note: V2 project keys (Token auth, /api/v2/task/sync) use a different
classifier-style API and are NOT compatible with V3 Playground keys.
"""
from __future__ import annotations

import base64
import io
import re
import time

import numpy as np

from app.config.settings import settings
from app.models.base import BaseDetector, ModelOutput

_DEFAULT_ENDPOINT = "https://api.thehive.ai/api/v3/chat/completions"
_MODEL_ID = "hive/vision-language-model"

# Carefully worded to elicit a bare decimal and nothing else.
_PROMPT = (
    "Examine this image for signs of being AI-generated, a GAN or diffusion model "
    "output, or a deepfake (face swap / face reenactment). Look for: unnaturally smooth "
    "skin, inconsistent lighting, blurred hair or teeth, colour bleeding at edges, "
    "implausible eye reflections, or perfect bilateral symmetry. "
    "Reply with ONLY a single decimal number between 0.00 and 1.00 — the probability "
    "that this image is AI-generated or fake. "
    "0.00 = definitely authentic.  1.00 = definitely AI-generated / fake. "
    "No other text."
)


def _parse_vlm_response(text: str) -> float | None:
    """Extract the first float in [0, 1] from the VLM reply."""
    text = text.strip()
    # Match any decimal/integer: '0.87', '.87', '87%' → 0.87
    matches = re.findall(r"\d+\.?\d*", text)
    for m in matches:
        v = float(m)
        if v > 1.0:
            v /= 100.0          # percentage form (e.g. "85" → 0.85)
        if 0.0 <= v <= 1.0:
            return v
    return None


class HiveDetector(BaseDetector):
    model_name = "hive"

    def __init__(self) -> None:
        self._loaded = False
        self._api_key: str = ""

    def load(self, weights_path: str, device) -> None:
        self._api_key = weights_path  # registry passes the api_key here
        if not self._api_key:
            raise ValueError("Hive API key is empty — set ext_api_key in settings.")
        endpoint = settings.ext_api_url or _DEFAULT_ENDPOINT
        print(f"✅ Hive VLM detector ready (key: …{self._api_key[-4:]}, endpoint: {endpoint})")
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, preprocessed: dict) -> ModelOutput:
        import requests

        t0 = time.time()
        pil = preprocessed["pil"]

        # Encode image as base64 JPEG
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=88)
        b64 = base64.b64encode(buf.getvalue()).decode()

        fake_prob = 0.5
        try:
            endpoint = settings.ext_api_url or _DEFAULT_ENDPOINT
            payload = {
                "model": _MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text",      "text": _PROMPT},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}"
                            }},
                        ],
                    }
                ],
                "max_tokens": 16,
            }
            resp = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30,
            )
            if not resp.ok:
                try:
                    msg = resp.json().get("message", resp.text[:200])
                except Exception:
                    msg = resp.text[:200]
                print(f"⚠️ Hive VLM {resp.status_code}: {msg} — using neutral score 0.5")
            else:
                content = resp.json()["choices"][0]["message"]["content"]
                parsed = _parse_vlm_response(content)
                if parsed is not None:
                    fake_prob = parsed
                else:
                    print(f"⚠️ Hive VLM: could not parse score from reply {content!r} — using 0.5")

        except Exception as exc:
            print(f"⚠️ Hive VLM call failed: {exc!r} — using neutral score 0.5")

        fake_prob = max(min(float(fake_prob), 0.999), 0.001)
        elapsed = (time.time() - t0) * 1000

        return ModelOutput(
            model_name=self.model_name,
            fake_prob=fake_prob,
            real_prob=1.0 - fake_prob,
            inference_time_ms=elapsed,
            features=np.array([fake_prob]),
        )
