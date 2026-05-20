"""
Per-model 1-D Platt scaling on the Validation set.

Fits sigmoid(a * raw_score + b) so each model's fake-probability is recentered
around 0.5. Writes calibration params to backend/app/models/weights/calibration.json.

Run once (or after retraining) — detectors auto-load this at startup.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.models.efficientnet import EfficientNetDetector
from app.models.f3net import _F3NetWrapper
from app.preprocessing.face_detection import detect_largest_face
from app.preprocessing.image_transforms import preprocess


def collect_eff_scores(model: EfficientNetDetector, paths: list[Path]) -> np.ndarray:
    """EfficientNet runs on face crop. Returns fake_prob in [0, 1]."""
    out: list[float] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        crop = detect_largest_face(np.array(img))
        view = Image.fromarray(crop) if crop is not None else img
        pre = preprocess(view)
        out.append(model.predict(pre).fake_prob)
    return np.array(out)


def collect_f3_logit_diff(model: _F3NetWrapper, paths: list[Path]) -> np.ndarray:
    """F3Net runs on full image at 256×256. Returns logit[0] − logit[1] (bigger → more fake)."""
    out: list[float] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        x = preprocess(img)["dfb_tensor"].unsqueeze(0)
        with torch.no_grad():
            logits = model(x)[0]
        out.append(float(logits[0] - logits[1]))
    return np.array(out)


def fit_platt(scores: np.ndarray, labels: np.ndarray, iters: int = 5000, lr: float = 0.1) -> tuple[float, float]:
    """1-D logistic regression: P(fake | x) = sigmoid(a * x + b). Returns (a, b)."""
    x = scores.astype(np.float64)
    y = labels.astype(np.float64)
    a, b = 0.0, 0.0
    for _ in range(iters):
        z = a * x + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_a = float(((p - y) * x).mean())
        grad_b = float((p - y).mean())
        a -= lr * grad_a
        b -= lr * grad_b
    return a, b


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    n = len(pos) * len(neg)
    if n == 0:
        return 0.5
    rank_sum = sum(1.0 if p > q else (0.5 if p == q else 0.0) for p in pos for q in neg)
    return rank_sum / n


def best_acc(scores_fake: np.ndarray, scores_real: np.ndarray, threshold: float = 0.5) -> float:
    correct = int((scores_fake > threshold).sum() + (scores_real <= threshold).sum())
    return correct / (len(scores_fake) + len(scores_real))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-root", type=Path, default=Path("/Users/user/Downloads/Dataset/Validation"))
    parser.add_argument("--n", type=int, default=400, help="Samples per class")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("app/models/weights/calibration.json"))
    args = parser.parse_args()

    random.seed(args.seed)

    real_paths = random.sample(sorted((args.val_root / "Real").glob("*.jpg")), args.n)
    fake_paths = random.sample(sorted((args.val_root / "Fake").glob("*.jpg")), args.n)

    # --- EfficientNet calibration ---
    print(f"Loading EfficientNet …")
    eff = EfficientNetDetector()
    eff.load("app/models/weights/efficientnet_binary.pth", torch.device("cpu"))
    print(f"Scoring {2 * args.n} validation images on EfficientNet (face crop)…")
    eff_fake = collect_eff_scores(eff, fake_paths)
    eff_real = collect_eff_scores(eff, real_paths)

    eff_auc = auc(eff_fake, eff_real)
    eff_raw_acc = best_acc(eff_fake, eff_real, 0.5)

    eff_scores = np.concatenate([eff_fake, eff_real])
    eff_labels = np.concatenate([np.ones(len(eff_fake)), np.zeros(len(eff_real))])
    eff_a, eff_b = fit_platt(eff_scores, eff_labels)
    eff_cal_fake = 1.0 / (1.0 + np.exp(-(eff_a * eff_fake + eff_b)))
    eff_cal_real = 1.0 / (1.0 + np.exp(-(eff_a * eff_real + eff_b)))
    eff_cal_acc = best_acc(eff_cal_fake, eff_cal_real, 0.5)

    print(f"  EfficientNet  AUC={eff_auc:.3f}  raw_acc={eff_raw_acc*100:.1f}%  cal_acc={eff_cal_acc*100:.1f}%")
    print(f"  EfficientNet  Platt: fake_prob → sigmoid({eff_a:.4f} * fake_prob + {eff_b:.4f})")

    # --- F3Net calibration ---
    print(f"Loading F3Net …")
    f3 = _F3NetWrapper()
    f3.load_state_dict(torch.load("app/models/weights/f3net_binary_best.pth", map_location="cpu", weights_only=False))
    f3.eval()
    print(f"Scoring {2 * args.n} validation images on F3Net (full image, logit space)…")
    f3_fake = collect_f3_logit_diff(f3, fake_paths)
    f3_real = collect_f3_logit_diff(f3, real_paths)

    f3_auc = auc(f3_fake, f3_real)
    # Default "raw acc" in fake_prob space (i.e. softmax[0] > 0.5)
    f3_raw_fake_probs = 1.0 / (1.0 + np.exp(-f3_fake))
    f3_raw_real_probs = 1.0 / (1.0 + np.exp(-f3_real))
    f3_raw_acc = best_acc(f3_raw_fake_probs, f3_raw_real_probs, 0.5)

    f3_scores = np.concatenate([f3_fake, f3_real])
    f3_labels = np.concatenate([np.ones(len(f3_fake)), np.zeros(len(f3_real))])
    f3_a, f3_b = fit_platt(f3_scores, f3_labels, lr=0.01)
    f3_cal_fake = 1.0 / (1.0 + np.exp(-(f3_a * f3_fake + f3_b)))
    f3_cal_real = 1.0 / (1.0 + np.exp(-(f3_a * f3_real + f3_b)))
    f3_cal_acc = best_acc(f3_cal_fake, f3_cal_real, 0.5)

    print(f"  F3Net         AUC={f3_auc:.3f}  raw_acc={f3_raw_acc*100:.1f}%  cal_acc={f3_cal_acc*100:.1f}%")
    print(f"  F3Net         Platt: logit_diff → sigmoid({f3_a:.4f} * logit_diff + {f3_b:.4f})")

    payload = {
        "efficientnet": {
            "domain": "fake_prob",
            "a": float(eff_a),
            "b": float(eff_b),
            "auc": float(eff_auc),
            "raw_acc": float(eff_raw_acc),
            "cal_acc": float(eff_cal_acc),
        },
        "f3net": {
            "domain": "logit_diff",
            "a": float(f3_a),
            "b": float(f3_b),
            "auc": float(f3_auc),
            "raw_acc": float(f3_raw_acc),
            "cal_acc": float(f3_cal_acc),
        },
        "_note": "Calibrated 1-D Platt scaling. Domain says whether `a*x+b` operates on raw fake_prob (efficientnet) or logit[0]-logit[1] (f3net).",
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote calibration to {args.output}")


if __name__ == "__main__":
    main()
