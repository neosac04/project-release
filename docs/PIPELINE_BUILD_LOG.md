# Deepfake Detection Pipeline — Build Log

> **Purpose**: Living document. Records every decision, finding, and change made session by session.
> If the chat window resets, read this file first — it has full context to resume immediately.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Original Requirements](#2-original-requirements)
3. [Architecture & Flow](#3-architecture--flow)
4. [Model & Weight Inventory](#4-model--weight-inventory)
5. [Dataset Inventory](#5-dataset-inventory)
6. [File Status Map](#6-file-status-map)
7. [Change Log](#7-change-log)
8. [Next 3 Steps](#8-next-3-steps)
9. [Key Design Decisions](#9-key-design-decisions)
10. [Known Bugs & Flags](#10-known-bugs--flags)

---

## 1. Project Overview

**What it is**: FastAPI backend for detecting deepfake / AI-generated images. Uses three models in parallel, a confidence-weighted fusion engine, GradCAM visual explanations, and a masked external API fallback.

**Goal**: 75–80% accuracy on diverse inputs. Hive Moderation API will be added as the masked fallback ("spectral_analyzer") once local models are wired and trained.

**Scope now**: Image detection only. Video detection is a later phase.

**Stack**: Python 3.11 · FastAPI · PyTorch · timm · MediaPipe · OpenCV · httpx · structlog

**Repo layout**:
```
FinalYrProj/
├── backend/
│   ├── app/
│   │   ├── models/weights/      ← all .pth files live here
│   │   ├── core/                ← pipeline, fusion
│   │   ├── models/              ← detector classes
│   │   ├── explainability/      ← gradcam, overlay
│   │   ├── preprocessing/       ← face detection, transforms
│   │   ├── api/endpoints/       ← detection, health, visualization
│   │   ├── schemas/             ← response models
│   │   ├── services/            ← external API client
│   │   ├── storage/             ← result cache
│   │   └── config/              ← settings
│   └── train_effnet_head.py     ← head-only EfficientNet training script
├── frontend/
└── docs/
    └── PIPELINE_BUILD_LOG.md    ← this file
```

**Backend entry**: `uvicorn app.main:app --host 0.0.0.0 --port 8000` (run from `backend/`)

---

## 2. Original Requirements

> Build a deepfake image classifier pipeline using EfficientNet, XceptionNet and F3Net.
> Flow: **Input image → face detection → model trio (parallel) → adaptive fusion → external API fallback (masked) → GradCAM explanations → final result**
>
> An external API (Hive Moderation) will be added as a fallback but its predictions must be shown as our model's prediction — masked as "spectral_analyzer" in the response.

---

## 3. Architecture & Flow

```
Input Image
    │
    ▼
Face Detection (MediaPipe → Haar fallback)
    │
    ├──► Full Image ──────────────────────────────────────┐
    │                                                     │
    └──► Face Crop ───────────────────────────────────────►│
                                                          ▼
                              ┌─────────────────────────────────────┐
                              │         Model Trio (parallel)       │
                              │  EfficientNet-B4   (face crop)      │
                              │  XceptionNet       (face crop)      │
                              │  F3Net             (full image)     │
                              └─────────────────────────────────────┘
                                                  │
                              ┌───────────────────▼─────────────────┐
                              │        Adaptive Fusion Engine        │
                              │   confidence-weighted average        │
                              │   + face-presence weight shift       │
                              │   uncertainty band: 0.38 – 0.62     │
                              └───────────────────────────────────────┘
                                                  │
                              ┌───────────────────▼─────────────────┐
                              │   External API Fallback (Hive)      │
                              │   masked as "spectral_analyzer"     │
                              │   called only when uncertain        │
                              └─────────────────────────────────────┘
                                                  │
                              ┌───────────────────▼─────────────────┐
                              │   GradCAM per model + ensemble      │
                              └─────────────────────────────────────┘
                                                  │
                                            Final Response
```

### Fusion Weights

| Condition | EfficientNet | XceptionNet | F3Net |
|-----------|-------------|-------------|-------|
| Face detected | 0.35 | 0.35 | 0.30 |
| No face | 0.20 | 0.20 | 0.60 |

Confidence bonus per model: `abs(fake_prob − 0.5) × 0.1`, then re-normalize.
Uncertainty band `0.38 ≤ score ≤ 0.62` → triggers Hive API call.
Hive result enters fusion at ~0.20 weight (4-model re-normalization). Cannot override a confident local verdict.

---

## 4. Model & Weight Inventory

### Weight Files — Full Picture

All weights live in `backend/app/models/weights/`.

| File | Size | Status | Head shape | What it is |
|------|------|--------|-----------|------------|
| `efficientnet-b4-6ed6700e.pth` | 74 MB | ✅ Present | `_fc: (1000, 1792)` | ImageNet pretrained, `efficientnet_pytorch` format. Used as backbone init for training. |
| `efficientnet_binary.pth` | — | ❌ Not yet created | `classifier.1: (2, 1792)` | Output of training script. Will be the active EfficientNet weight. |
| `xception-b5690688.pth` | 87 MB | 🗑️ Superseded | `fc: (1000, 2048)` | Old ImageNet pretrained xception (timm flat format). Replace with `xception_best.pth`. |
| `xception_best.pth` | — | ⚠️ Downloaded, not copied yet | `backbone.last_linear: (2, 2048)` | **Deepfake-trained** xception. `backbone.*` prefix. Has extra `adjust_channel` 1×1 conv. Copy from `/Users/nsac/Downloads/xception_best.pth`. |
| `f3net_best.pth` | 86 MB | ✅ Present | `backbone.last_linear.1: (2, 2048)` | **Deepfake-trained** F3Net. Real paper architecture with SRM input + FAD head. Ready to use. |

### Model Architecture Details

**EfficientNet-B4**
- Library: `torchvision` (not `efficientnet_pytorch`)
- Input: 380×380 RGB, ImageNet normalisation
- Training: freeze backbone, train `classifier[1]` (2-class head) only
- After training: `classifier.1.weight` shape `(2, 1792)`
- Class order from `ImageFolder` on `Dataset/`: `Fake=0, Real=1`
- ⚠️ Bug: current `efficientnet.py` uses `probs[1]` for fake_prob — must change to `probs[0]` after training

**XceptionNet**
- Format: `backbone.*` prefix, same codebase as F3Net (DeepfakeBench)
- Input: 299×299 RGB, ImageNet normalisation
- Extra layer: `backbone.adjust_channel` — 1×1 Conv2d(2048→512) + BN. Must be in model class or load fails.
- GradCAM target: `backbone.bn4`
- Head: `backbone.last_linear.weight (2, 2048)` — binary, Fake=0 Real=1

**F3Net**
- Architecture: NOT a simple ResNet34+DCT (original plan was wrong)
- Backbone: Xception with **12-channel SRM input** (`backbone.conv1.weight: (32, 12, 3, 3)`)
  - Requires SRM preprocessing: 3-channel RGB → 12-channel SRM residuals before backbone
- FAD head: `FAD_head.*` — 4 learnable DCT filter banks (256×256 each), frequency-aware decomposition
- Head: `backbone.last_linear.1.weight (2, 2048)` — binary, Fake=0 Real=1
- Input: Full image (not face crop) — frequency artefacts are global
- GradCAM target: TBD (need to verify after implementation)

---

## 5. Dataset Inventory

All datasets at `/Users/nsac/Downloads/Datasets/`.

| Folder | Real | Fake | Size | Format | Verdict |
|--------|------|------|------|--------|---------|
| `Dataset/Train` | 70,001 | 70,001 | 256×256 | JPG | ✅ **USE THIS** |
| `Dataset/Validation` | 19,787 | 19,641 | 256×256 | JPG | ✅ Already split |
| `Dataset/Test` | 5,413 | 5,492 | 256×256 | JPG | ✅ For post-train eval |
| `archive/real_and_fake_face` | 1,081 | 960 | 600×600 | JPG | ⛔ Too small (~2K total) |
| `archive (2)/train` | 20,699 | 73,154 | 160×160 | PNG | ⛔ 3.5× imbalanced + too low res |
| `cropped_images/` | unlabeled | unlabeled | 150×150 | PNG | ⛔ No class labels, needs reorganisation |
| `deepfake-vs-real-60k/` | 28,475 | 28,596 | 512×512 vs variable | PNG vs JPG | ⛔ Format mismatch — real=face crops, fake=AI portrait photos |

**Why `Dataset` wins**: Both classes are 256×256 JPG face crops, perfectly balanced, pre-split into Train/Val/Test. No spurious format signals. EfficientNet script maps directly with no reorganisation needed.

---

## 6. File Status Map

### Files That Exist

| File | Status | Notes |
|------|--------|-------|
| `backend/app/models/base.py` | ✅ Working | `ModelOutput`, `BaseDetector` ABC |
| `backend/app/models/efficientnet.py` | ⚠️ Needs fix | GradCAM++ wired. Bug: uses `probs[1]` for fake — must be `probs[0]` after training |
| `backend/app/models/univfd.py` | 🗑️ De-register | Will stay on disk but removed from registry |
| `backend/app/models/registry.py` | ⚠️ Outdated | Only loads univfd + efficientnet. Needs xception + f3net |
| `backend/app/core/pipeline.py` | ⚠️ Rewrite | 2-model, hard-coded 60/40. Full rewrite needed |
| `backend/app/schemas/response.py` | ⚠️ Rewrite | Has `univfd_score`, `efficientnet_score`. Needs `model_votes` dict |
| `backend/app/config/settings.py` | ⚠️ Update | Missing `ext_api_url`, `ext_api_key`, uncertainty bounds |
| `backend/app/explainability/overlay.py` | ✅ Working | Heatmap overlay + ensemble blend |
| `backend/app/preprocessing/face_detection.py` | ✅ Working | MediaPipe + Haar fallback |
| `backend/app/preprocessing/image_transforms.py` | ✅ Working | ImageNet + freq transforms |
| `backend/app/storage/result_cache.py` | ✅ Working | Thread-safe LRU, stores image bytes |
| `backend/app/api/endpoints/detection.py` | ⚠️ Update | Must pass `image_bytes` into pipeline |
| `backend/app/api/endpoints/visualization.py` | ⚠️ Update | Hard-coded to univfd/efficientnet model names |
| `backend/app/api/endpoints/health.py` | ⚠️ Update | Checks for `univfd` by name — update to trio |
| `backend/train_effnet_head.py` | ✅ Ready to run | Head-only EfficientNet training. Never been run yet. |

### Files to Create from Scratch

| File | Purpose |
|------|---------|
| `backend/app/explainability/gradcam.py` | Unified hook-based GradCAM for XceptionNet + F3Net |
| `backend/app/models/xceptionnet.py` | XceptionNet detector with `backbone.*` architecture |
| `backend/app/models/f3net.py` | F3Net detector with SRM preprocessing + FAD head |
| `backend/app/core/fusion.py` | Adaptive confidence-weighted fusion engine |
| `backend/app/services/external_api.py` | Hive API client masked as "spectral_analyzer" |

---

## 7. Change Log

### Session 1 — 2026-05-11

**What we did**: Full codebase audit. Defined architecture. No code written yet.

**Key findings**:
- Pipeline currently runs on EfficientNet (broken — ImageNet weights, 1000-class) + UnivFD (CLIP-based, wrong domain). Effectively only one meaningful model exists.
- `train_effnet_head.py` was written but never run — `efficientnet_binary.pth` doesn't exist.
- `f3net_best.pth` is the only fully deepfake-trained weight in the project.

---

### Session 2 — 2026-05-11

**What we did**: Inspected all weight files. Evaluated new xception weight. Evaluated 5 training datasets. Determined training plan.

**Weight file discoveries**:

1. **`efficientnet_b4.pth`** (worktree active file):
   - `classifier.1.weight: (1000, 1792)` — ImageNet pretrained, torchvision format
   - NOT fine-tuned. Pipeline using this is outputting the "goldfish" class probability as fake_prob. Completely wrong.

2. **`efficientnet-b4-6ed6700e.pth`** (weights folder):
   - Same: ImageNet pretrained, `efficientnet_pytorch` format (`_conv_stem`, `_blocks`, `_fc`)
   - `_fc.weight: (1000, 1792)` — 1000-class
   - Use as backbone init for training, not for inference

3. **`xception-b5690688.pth`** (weights folder — OLD):
   - `fc.weight: (1000, 2048)` — ImageNet pretrained, timm flat format
   - Superseded by `xception_best.pth`

4. **`xception_best.pth`** (downloaded to `/Users/nsac/Downloads/`):
   - `backbone.last_linear.weight: (2, 2048)` — **binary deepfake classifier**
   - `backbone.*` prefix wrapping entire network
   - Extra `backbone.adjust_channel.*` layer: Conv2d(2048→512, 1×1) + BN
   - Same `backbone.*` key format as `f3net_best.pth` — same research codebase (DeepfakeBench)
   - 3-channel RGB input (no SRM) — same conv1 shape as old file
   - **This is the weight to use for XceptionNet**

5. **`f3net_best.pth`** (weights folder):
   - `backbone.conv1.weight: (32, 12, 3, 3)` — **12-channel SRM input**, not standard 3-channel RGB
   - `FAD_head.*` — 4 learnable DCT filter banks (256×256), frequency-aware decomposition
   - `backbone.last_linear.1.weight: (2, 2048)` — binary deepfake classifier
   - Full F3Net paper architecture (much more complex than original plan assumed)
   - **Already trained. Ready to use once correctly implemented.**

**F3Net architecture correction** (original plan was wrong):
- Original plan: ResNet34 RGB + DCT branch
- Reality: Xception backbone with 12-channel SRM preprocessed input + FAD (Frequency-Aware Decomposition) head
- Impact: SRM filter preprocessing step needed before backbone (turns 3-ch RGB → 12-ch residuals)

**Dataset evaluation** (5 datasets inspected):
- Eliminated `deepfake-vs-real-60k` (512×512 PNG real vs portrait JPG fake — format mismatch, model learns format not deepfake artifacts)
- Eliminated `archive` (only ~2K images total, too small)
- Eliminated `archive (2)` (3.5× class imbalance + 160×160 too small for EfficientNet at 380×380)
- Eliminated `cropped_images` (no class labels in folder structure, 150×150 too small)
- **Selected `Dataset`**: 70K×70K balanced, both 256×256 JPG, pre-split Train/Val/Test, direct drop-in for training script

**Training plan finalised**: head-only (freeze backbone, train `classifier[1]` only). Script `train_effnet_head.py` already written and ready. Dataset maps directly — no reorganisation needed.

---

## 8. Next 3 Steps

### Step 1 — Train EfficientNet ← DO THIS NOW

**What**: Run the existing training script against the `Dataset` folder. Head-only: ImageNet backbone frozen, only the final 2-class linear layer is trained.

**Command**:
```bash
cd /Users/nsac/Projects/FinalYrProj/FinalYrProj/backend

python train_effnet_head.py \
  --dataset-root /Users/nsac/Downloads/Datasets/Dataset \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-3 \
  --output ./app/models/weights/efficientnet_binary.pth
```

**What to watch for**:
- Script prints `Class mapping:` at start — confirm it shows `{'Fake': 0, 'Real': 1}`
- Val accuracy should climb each epoch. Expect 70–80% by epoch 10 on this dataset.
- Best checkpoint is auto-saved to `./app/models/weights/efficientnet_binary.pth`

**Verify after training**:
```bash
python3 -c "
import torch
ck = torch.load('app/models/weights/efficientnet_binary.pth', map_location='cpu', weights_only=False)
print('head shape:', tuple(ck['classifier.1.weight'].shape))
# Must print: (2, 1792)
"
```

**Report back**: val accuracy achieved + class mapping printed by script.

---

### Step 2 — Copy `xception_best.pth` into project weights folder

**What**: The new XceptionNet weights are sitting in Downloads. Move them into the project.

**Command**:
```bash
cp /Users/nsac/Downloads/xception_best.pth \
   /Users/nsac/Projects/FinalYrProj/FinalYrProj/backend/app/models/weights/xception_best.pth
```

The old `xception-b5690688.pth` can stay (keep for reference) but will not be registered in the pipeline.

---

### Step 3 — Implement `app/models/xceptionnet.py`

**What**: Build the XceptionNet detector class that loads `xception_best.pth` correctly.

**Architecture requirements** (from checkpoint inspection):
- Keys all prefixed with `backbone.*`
- Blocks: `backbone.block1` – `backbone.block12` (standard Xception depthwise separable blocks)
- Exit flow: `backbone.conv3`, `backbone.conv4`, `backbone.bn3`, `backbone.bn4`
- Extra layer: `backbone.adjust_channel` — `Conv2d(2048, 512, kernel_size=1)` + `BatchNorm2d(512)`
  - Used in DeepfakeBench for feature distillation; must be in model or 7 keys fail to load
- Classifier: `backbone.last_linear.weight (2, 2048)` + bias
- Input: 3-channel RGB, 299×299, ImageNet normalisation
- GradCAM target layer: `backbone.bn4`

**Load strategy**: build model with `backbone.*` structure, call `model.load_state_dict(checkpoint, strict=True)`. All 283 keys should load cleanly.

**predict()**: `softmax(logits)[0]` = fake_prob (Fake=0 in class order).

**get_heatmap()**: call `gradcam(self.model, "backbone.bn4", tensor, device)` — requires `gradcam.py` to exist first (build that before or alongside this).

---

## 9. Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Replace UnivFD | ✅ Yes — with F3Net | CLIP backbone is wrong domain for deepfakes; F3Net has frequency-domain complementarity |
| EfficientNet head training | Head-only (frozen backbone) | Fast (30–60 min CPU), sufficient for a proof-of-concept; full fine-tuning needs GPU + days |
| Training dataset | `Dataset/` (256×256 JPG, 140K balanced) | Only dataset where both classes share the same format and size — eliminates spurious format signals |
| XceptionNet weight | `xception_best.pth` not `xception-b5690688.pth` | New file is deepfake-trained (binary head); old is ImageNet (1000-class, useless for deepfakes) |
| F3Net input | Full image, not face crop | Frequency artefacts are global; compression DCT grid doesn't align with face region |
| F3Net preprocessing | SRM filters: 3-ch RGB → 12-ch residuals | Required by `f3net_best.pth` — `backbone.conv1` expects 12 channels |
| GradCAM variant | GradCAM++ for EfficientNet, standard GradCAM for Xception + F3Net | EfficientNet already has GradCAM++ wired; standard GradCAM sufficient for others |
| External API strategy | Call only when `0.38 ≤ score ≤ 0.62` | Preserves low latency for clear-cut cases; Hive API only needed for genuinely uncertain ones |
| API masking key | `"spectral_analyzer"` | Blends naturally in `model_votes` dict; frontend never sees the word "external" |
| Hive API weight | ~0.20 after 4-model re-normalization | Cannot reverse a confident local verdict; nudges uncertain cases only |

---

## 10. Known Bugs & Flags

| Item | Severity | Status | What to do |
|------|----------|--------|------------|
| `efficientnet.py` uses `probs[1]` for fake_prob | 🔴 High | Open | After training on `Dataset/`, class order is `Fake=0, Real=1`. Must change to `probs[0]`. Fix during Step 3 wire-up. |
| `efficientnet_b4.pth` (worktree active) is ImageNet 1000-class | 🔴 High | Will be fixed by Step 1 | Training produces `efficientnet_binary.pth` with 2-class head |
| `xception_best.pth` not yet in project | 🟡 Medium | Fix in Step 2 | Copy from Downloads |
| F3Net needs SRM preprocessing (12-channel input) | 🟡 Medium | Open | Must implement SRM filter layer before backbone in `f3net.py` |
| `app/core/uncertainty.py` referenced in old tests but doesn't exist | 🟡 Medium | Open | Build alongside or after fusion engine |
| Hive API URL/key not configured | 🟡 Medium | Open | Add `EXT_API_URL` + `EXT_API_KEY` to `.env` when ready to integrate |
| Frontend uses `univfd_score` + `efficientnet_score` fields | 🟡 Medium | Open | Schema rewrite will remove these — frontend must migrate to `model_votes` dict |
| `xception-b5690688.pth` still in weights folder | 🟢 Low | After Step 2 | Can delete or keep for reference — just don't register it |

---

*Last updated: 2026-05-11 — Session 2 complete. Weight inspection done. Dataset selected. Training plan finalised. No code written yet.*
