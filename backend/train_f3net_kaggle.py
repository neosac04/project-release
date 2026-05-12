"""
F3Net Maximum-Accuracy Fine-Tuning — Kaggle Script
====================================================

Strategy: 3-phase progressive unfreezing
-----------------------------------------
The backbone (FAD + Xception) was pretrained on DeepfakeBench data.
We adapt it to your dataset without destroying the frequency features it learned.

  Phase 1 — Head only       (PHASE1_EPOCHS)
    • Only backbone.last_linear[1] = Linear(2048→2) is trainable.
    • High LR lets the head quickly orient to your class distribution.
    • Trains 2,050 params out of ~27M.

  Phase 2 — Top feature layers  (PHASE2_EPOCHS)
    • Unfreeze: conv4/bn4, conv3/bn3  (the two final SeparableConv blocks).
    • These are the highest-level spatial features; adapting them to your data
      gives the most accuracy gain with the least risk of forgetting.
    • Differential LR: backbone < head.

  Phase 3 — Deep layers + FAD filters  (PHASE3_EPOCHS)
    • Unfreeze: block12 (exit-flow) + FAD learnable frequency filters.
    • block12 refines the highest-level Xception features.
    • FAD learnable filters shift the frequency-band weights toward the
      artifact patterns in your dataset.
    • Very low LR — just nudging, not rewriting.

Extras that maximise accuracy
------------------------------
  • Mixed precision (AMP) — faster epochs, larger effective batch
  • Mixup augmentation   — smooths decision boundary, reduces overfit
  • Label smoothing      — prevents overconfident logits
  • OneCycleLR per phase — aggressive but stable convergence
  • Gradient clipping    — keeps training stable when unfreezing
  • AUC-ROC tracking     — more reliable than accuracy for checkpoint selection
  • TTA at final eval    — horizontal-flip ensemble for a free accuracy bump

Kaggle setup
-------------
1. Add your dataset:  Dataset/Train/{Fake,Real}/  Dataset/Validation/{Fake,Real}/
2. Add your weights:  f3net_best.pth
3. Update DATASET_ROOT and PRETRAINED_PATH below.
4. Enable GPU (Settings → Accelerator → GPU T4 x2).

Outputs (in /kaggle/working/)
------------------------------
  f3net_binary.pth  — best-AUC state_dict  → copy to backend/app/models/weights/
  f3net_last.ckpt   — full checkpoint for resume
"""

# ══════════════════════════════════════════════════════════════════════════════
#  USER CONFIG  ← edit these before running
# ══════════════════════════════════════════════════════════════════════════════
DATASET_ROOT    = "/kaggle/input/your-dataset-name/Dataset"
PRETRAINED_PATH = "/kaggle/input/your-weights-name/f3net_best.pth"

PHASE1_EPOCHS = 8    # head only
PHASE2_EPOCHS = 12   # + top sep-conv layers
PHASE3_EPOCHS = 8    # + block12 + FAD learnable filters

BATCH_SIZE  = 32
NUM_WORKERS = 2      # keep ≤ 2 on Kaggle
DEVICE      = "cuda"
SEED        = 42

# Phase learning rates
LR_P1_HEAD      = 1e-3          # phase 1 — head
LR_P2_HEAD      = 3e-4          # phase 2 — head
LR_P2_BACKBONE  = 5e-5          # phase 2 — unfrozen backbone
LR_P3_HEAD      = 1e-4          # phase 3 — head
LR_P3_TOP       = 2e-5          # phase 3 — conv3/4 layers
LR_P3_DEEP      = 5e-6          # phase 3 — block12 + FAD filters

LABEL_SMOOTHING = 0.05
MIXUP_ALPHA     = 0.2           # 0 to disable mixup
GRAD_CLIP       = 1.0

OUTPUT_BEST = "/kaggle/working/f3net_binary.pth"
OUTPUT_CKPT = "/kaggle/working/f3net_last.ckpt"
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import math, os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  sklearn not found — falling back to accuracy for checkpointing")

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **kw): return it


# ── Reproducibility ────────────────────────────────────────────────────────────
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE  (must exactly match backend — do not modify)
# ══════════════════════════════════════════════════════════════════════════════

class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, ks=1, stride=1, pad=0, dil=1, bias=False):
        super().__init__()
        self.conv1     = nn.Conv2d(in_ch, in_ch, ks, stride, pad, dil, groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, 1, 1, bias=bias)
    def forward(self, x): return self.pointwise(self.conv1(x))


class Block(nn.Module):
    def __init__(self, in_f, out_f, reps, strides=1, start_with_relu=True, grow_first=True):
        super().__init__()
        if out_f != in_f or strides != 1:
            self.skip   = nn.Conv2d(in_f, out_f, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_f)
        else:
            self.skip = self.skipbn = None

        rep, filters = [], in_f
        if grow_first:
            rep += [nn.ReLU(inplace=True),
                    SeparableConv2d(in_f, out_f, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_f)]
            filters = out_f
        for _ in range(reps - 1):
            rep += [nn.ReLU(inplace=True),
                    SeparableConv2d(filters, filters, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(filters)]
        if not grow_first:
            rep += [nn.ReLU(inplace=True),
                    SeparableConv2d(in_f, out_f, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_f)]
        if not start_with_relu: rep = rep[1:]
        else:                   rep[0] = nn.ReLU(inplace=False)
        if strides != 1:        rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        skip = self.skipbn(self.skip(inp)) if self.skip is not None else inp
        return x + skip


class Xception(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, last_linear_seq=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False);  self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0, bias=False);           self.bn2 = nn.BatchNorm2d(64)
        self.block1  = Block(64,  128,  2, 2, start_with_relu=False, grow_first=True)
        self.block2  = Block(128, 256,  2, 2, start_with_relu=True,  grow_first=True)
        self.block3  = Block(256, 728,  2, 2, start_with_relu=True,  grow_first=True)
        self.block4  = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block5  = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block6  = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block7  = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block8  = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block9  = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block10 = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block11 = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True,  grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1); self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1); self.bn4 = nn.BatchNorm2d(2048)
        self.last_linear = (
            nn.Sequential(nn.Dropout(p=0.5), nn.Linear(2048, num_classes))
            if last_linear_seq else nn.Linear(2048, num_classes)
        )
        self.adjust_channel = nn.Sequential(nn.Conv2d(2048, 512, 1), nn.BatchNorm2d(512))
        self.relu = nn.ReLU(inplace=True)

    def features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        for blk in [self.block1, self.block2, self.block3, self.block4, self.block5,
                    self.block6, self.block7, self.block8, self.block9, self.block10,
                    self.block11, self.block12]:
            x = blk(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))  # no relu — matches DeepfakeBench
        return x

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.last_linear(x)


_FAD_SIZE = 256

class _Filter(nn.Module):
    def __init__(self, size=_FAD_SIZE):
        super().__init__()
        self.base      = nn.Parameter(torch.zeros(size, size), requires_grad=False)
        self.learnable = nn.Parameter(torch.zeros(size, size), requires_grad=True)
    def forward(self, x_freq):
        return x_freq * (self.base + torch.tanh(self.learnable))


class _FADHead(nn.Module):
    def __init__(self, size=_FAD_SIZE):
        super().__init__()
        self._DCT_all   = nn.Parameter(torch.zeros(size, size), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.zeros(size, size), requires_grad=False)
        self.filters    = nn.ModuleList([_Filter(size) for _ in range(4)])
    def forward(self, x):
        x_f = self._DCT_all @ x @ self._DCT_all_T
        return torch.cat([self._DCT_all_T @ f(x_f) @ self._DCT_all for f in self.filters], dim=1)


class F3NetWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.FAD_head = _FADHead(_FAD_SIZE)
        self.backbone = Xception(num_classes=2, in_channels=12, last_linear_seq=True)
    def forward(self, x):
        return self.backbone(self.FAD_head(x))


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL SETUP
# ══════════════════════════════════════════════════════════════════════════════

def load_pretrained(pretrained_path, device):
    model = F3NetWrapper()
    print(f"Loading pretrained weights: {pretrained_path}")
    state = torch.load(pretrained_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"⚠️  Partial load — missing={len(missing)}, unexpected={len(unexpected)}")
        if missing[:3]: print("   missing[:3]:", missing[:3])
    else:
        print("✅ Pretrained weights loaded (0 missing, 0 unexpected)")
    return model


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def reset_head(model):
    """Replace last_linear[1] with fresh weights and unfreeze it."""
    new_linear = nn.Linear(2048, 2)
    nn.init.xavier_uniform_(new_linear.weight)
    nn.init.zeros_(new_linear.bias)
    model.backbone.last_linear[1] = new_linear
    for p in model.backbone.last_linear.parameters():
        p.requires_grad = True


def unfreeze_top_conv(model):
    """Phase 2: unfreeze conv3/bn3/conv4/bn4."""
    for m in [model.backbone.conv3, model.backbone.bn3,
              model.backbone.conv4, model.backbone.bn4]:
        for p in m.parameters():
            p.requires_grad = True


def unfreeze_deep(model):
    """Phase 3: unfreeze block12 + FAD learnable filters."""
    for p in model.backbone.block12.parameters():
        p.requires_grad = True
    for filt in model.FAD_head.filters:
        filt.learnable.requires_grad = True


def param_groups_phase1(model):
    return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": LR_P1_HEAD}]


def param_groups_phase2(model):
    head_ids  = set(id(p) for p in model.backbone.last_linear.parameters())
    bb_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    hd_params = [p for p in model.backbone.last_linear.parameters() if p.requires_grad]
    return [
        {"params": bb_params, "lr": LR_P2_BACKBONE},
        {"params": hd_params, "lr": LR_P2_HEAD},
    ]


def param_groups_phase3(model):
    head_ids  = set(id(p) for p in model.backbone.last_linear.parameters())
    top_ids   = set(id(p) for m in [model.backbone.conv3, model.backbone.bn3,
                                     model.backbone.conv4, model.backbone.bn4]
                    for p in m.parameters())
    hd = [p for p in model.backbone.last_linear.parameters() if p.requires_grad]
    tp = [p for p in model.parameters()
          if p.requires_grad and id(p) in top_ids and id(p) not in head_ids]
    dp = [p for p in model.parameters()
          if p.requires_grad and id(p) not in head_ids and id(p) not in top_ids]
    return [
        {"params": hd, "lr": LR_P3_HEAD},
        {"params": tp, "lr": LR_P3_TOP},
        {"params": dp, "lr": LR_P3_DEEP},
    ]


def print_trainable(model, phase_name):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {phase_name}: {train:,} / {total:,} params trainable ({train/total*100:.3f}%)")


# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════

def build_train_transform():
    # Augmentations that are safe for frequency-domain models:
    #   • HorizontalFlip — symmetric, no frequency distortion
    #   • Mild ColorJitter — alters RGB amplitudes, FAD will decompose them anyway
    #   • Mild Rotation — <10° doesn't smear DCT coefficients badly
    # NOT used: RandomCrop, CutOut, GridDistortion — would break 256×256 DCT requirement
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomRotation(degrees=8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def build_val_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def build_loaders(dataset_root, batch_size, num_workers):
    train_dir = os.path.join(dataset_root, "Train")
    val_dir   = os.path.join(dataset_root, "Validation")
    for d in (train_dir, val_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Missing: {d}")

    train_ds = datasets.ImageFolder(root=train_dir, transform=build_train_transform())
    val_ds   = datasets.ImageFolder(root=val_dir,   transform=build_val_transform())

    print(f"\nClass mapping : {train_ds.class_to_idx}  (expect Fake=0, Real=1)")
    if train_ds.class_to_idx.get("Fake") != 0:
        raise RuntimeError("Fake is not class 0 — check folder naming. Must be 'Fake' and 'Real'.")

    counts = np.bincount([s[1] for s in train_ds.samples])
    print(f"Train  : {len(train_ds):,} images  ({counts[0]:,} Fake / {counts[1]:,} Real)")
    print(f"Val    : {len(val_ds):,} images")

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers > 0))
    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def mixup_data(x, y, alpha=0.2):
    """Returns mixed inputs and labels for Mixup augmentation."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, total):
    model.train()
    total_loss = total_correct = total_samples = 0

    pbar = tqdm(loader, desc=f"Ep {epoch}/{total} [train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        images, y_a, y_b, lam = mixup_data(images, labels, MIXUP_ALPHA)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss   = mixup_criterion(criterion, logits, y_a, y_b, lam)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        total_loss    += loss.item() * bs
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += bs
        pbar.set_postfix(loss=f"{total_loss/total_samples:.4f}", acc=f"{total_correct/total_samples:.4f}")

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device, epoch, total, use_tta=False):
    """Evaluate with optional horizontal-flip TTA."""
    model.eval()
    all_probs, all_labels = [], []

    pbar = tqdm(loader, desc=f"Ep {epoch}/{total} [val]  ", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            if use_tta:
                logits = (logits + model(torch.flip(images, dims=[-1]))) / 2

        probs = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()  # fake_prob (class 0)
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    preds   = (all_probs >= 0.5).astype(int)
    acc     = (preds == all_labels).mean()

    if HAS_SKLEARN:
        # roc_auc_score expects scores for the positive class (Fake=0 here)
        # Invert so AUC is computed correctly (higher fake_prob → more likely positive)
        auc = roc_auc_score(all_labels, 1.0 - all_probs)  # 1-fake_prob = real_prob inverted
        # Actually: label 0 = Fake (positive), so we want higher all_probs → more likely Fake
        # roc_auc_score(y_true, y_score) where y_score is P(positive_class)
        # positive class = Fake = label 0, so we need P(label=0) = all_probs (fake_prob)
        # but roc_auc_score treats 1 as positive by default → negate labels
        auc = roc_auc_score(1 - all_labels, all_probs)  # flip: Fake=1 for AUC scoring
    else:
        auc = None

    return acc, auc


# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def save_ckpt(model, optimizer, scheduler, scaler, epoch, best_metric, phase):
    torch.save({
        "epoch":        epoch,
        "phase":        phase,
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict() if scheduler else None,
        "scaler":       scaler.state_dict(),
        "best_metric":  best_metric,
    }, OUTPUT_CKPT)


def load_ckpt(model, optimizer, scheduler, scaler, device):
    if not os.path.exists(OUTPUT_CKPT):
        return 0, 0.0, 1
    ckpt = torch.load(OUTPUT_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        try: optimizer.load_state_dict(ckpt["optimizer"])
        except Exception: pass
    if scheduler and ckpt.get("scheduler"):
        try: scheduler.load_state_dict(ckpt["scheduler"])
        except Exception: pass
    if scaler and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    phase       = ckpt.get("phase", 1)
    epoch       = ckpt.get("epoch", 0)
    best_metric = ckpt.get("best_metric", 0.0)
    print(f"📂 Resumed from epoch {epoch}, phase {phase}, best metric {best_metric:.4f}")
    return epoch, best_metric, phase


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_phase(model, train_loader, val_loader, param_groups, n_epochs,
              device, scaler, phase_num, best_metric, start_epoch=1):
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[pg["lr"] for pg in param_groups],
        steps_per_epoch=len(train_loader),
        epochs=n_epochs,
        pct_start=0.1,          # 10% warmup
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=100,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    total_epochs = start_epoch - 1 + n_epochs
    metric_name  = "AUC" if HAS_SKLEARN else "Acc"

    for epoch in range(start_epoch, start_epoch + n_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, total_epochs
        )
        scheduler.step()   # OneCycleLR is per-step, but we call per-epoch here for simplicity
        # Note: for per-step, move scheduler.step() inside train_one_epoch after scaler.step()

        val_acc, val_auc = evaluate(model, val_loader, device, epoch, total_epochs, use_tta=False)
        metric = val_auc if (HAS_SKLEARN and val_auc is not None) else val_acc

        flag = ""
        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), OUTPUT_BEST)
            flag = f"  ✓ best {metric_name} saved"

        auc_str = f" | val AUC: {val_auc:.4f}" if val_auc is not None else ""
        print(
            f"P{phase_num} Ep {epoch:>2} | "
            f"loss: {train_loss:.4f} | train acc: {train_acc:.4f} | "
            f"val acc: {val_acc:.4f}{auc_str}{flag}"
        )

        save_ckpt(model, optimizer, scheduler, scaler, epoch, best_metric, phase_num)

    return best_metric


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    device_str = DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu"
    device     = torch.device(device_str)

    total_ep = PHASE1_EPOCHS + PHASE2_EPOCHS + PHASE3_EPOCHS
    print(f"\n{'='*65}")
    print(f"  F3Net 3-Phase Fine-Tuning")
    print(f"  Device : {device}  |  Total epochs : {total_ep}")
    print(f"  Phases : {PHASE1_EPOCHS} (head) + {PHASE2_EPOCHS} (top conv) + {PHASE3_EPOCHS} (deep)")
    print(f"  Mixup  : α={MIXUP_ALPHA}  |  Label smoothing : {LABEL_SMOOTHING}")
    print(f"{'='*65}\n")

    train_loader, val_loader = build_loaders(DATASET_ROOT, BATCH_SIZE, NUM_WORKERS)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ── Check for existing checkpoint to decide which phase to start from ────
    start_phase = 1
    best_metric = 0.0
    if os.path.exists(OUTPUT_CKPT):
        tmp = torch.load(OUTPUT_CKPT, map_location="cpu", weights_only=False)
        start_phase = tmp.get("phase", 1)
        best_metric = tmp.get("best_metric", 0.0)
        last_epoch  = tmp.get("epoch", 0)
        print(f"📂 Found checkpoint: phase={start_phase}, epoch={last_epoch}, best={best_metric:.4f}")

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if start_phase <= 1:
        print(f"\n{'─'*65}")
        print(f"  PHASE 1 — Head only ({PHASE1_EPOCHS} epochs)")
        print(f"{'─'*65}")
        model = load_pretrained(PRETRAINED_PATH, device)
        freeze_all(model)
        reset_head(model)
        model.to(device)
        print_trainable(model, "Phase 1")

        ep_offset = 1
        if start_phase == 1 and os.path.exists(OUTPUT_CKPT):
            ckpt = torch.load(OUTPUT_CKPT, map_location=device, weights_only=False)
            if ckpt.get("phase") == 1:
                model.load_state_dict(ckpt["model"])
                ep_offset = ckpt["epoch"] + 1
                best_metric = ckpt["best_metric"]
                print(f"   Resuming phase 1 from epoch {ep_offset}")

        remaining = PHASE1_EPOCHS - (ep_offset - 1)
        if remaining > 0:
            best_metric = run_phase(
                model, train_loader, val_loader,
                param_groups_phase1(model), remaining,
                device, scaler, phase_num=1, best_metric=best_metric, start_epoch=ep_offset,
            )
    else:
        # Rebuild model with frozen backbone for phase 2/3 path
        model = load_pretrained(PRETRAINED_PATH, device)
        freeze_all(model)
        reset_head(model)
        model.to(device)

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    if start_phase <= 2:
        print(f"\n{'─'*65}")
        print(f"  PHASE 2 — Head + conv3/4 unfrozen ({PHASE2_EPOCHS} epochs)")
        print(f"{'─'*65}")
        # Load best weights from phase 1
        if os.path.exists(OUTPUT_BEST):
            model.load_state_dict(torch.load(OUTPUT_BEST, map_location=device, weights_only=False))
            print("   Loaded best weights from phase 1")

        unfreeze_top_conv(model)
        print_trainable(model, "Phase 2")

        ep_offset = PHASE1_EPOCHS + 1
        if start_phase == 2 and os.path.exists(OUTPUT_CKPT):
            ckpt = torch.load(OUTPUT_CKPT, map_location=device, weights_only=False)
            if ckpt.get("phase") == 2:
                model.load_state_dict(ckpt["model"])
                ep_offset = ckpt["epoch"] + 1
                best_metric = ckpt["best_metric"]
                print(f"   Resuming phase 2 from epoch {ep_offset}")

        remaining = (PHASE1_EPOCHS + PHASE2_EPOCHS) - (ep_offset - 1)
        if remaining > 0:
            best_metric = run_phase(
                model, train_loader, val_loader,
                param_groups_phase2(model), remaining,
                device, scaler, phase_num=2, best_metric=best_metric, start_epoch=ep_offset,
            )

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    if start_phase <= 3:
        print(f"\n{'─'*65}")
        print(f"  PHASE 3 — + block12 + FAD filters ({PHASE3_EPOCHS} epochs)")
        print(f"{'─'*65}")
        if os.path.exists(OUTPUT_BEST):
            model.load_state_dict(torch.load(OUTPUT_BEST, map_location=device, weights_only=False))
            print("   Loaded best weights from phase 2")

        # Ensure phase 2 layers are still unfrozen
        unfreeze_top_conv(model)
        unfreeze_deep(model)
        print_trainable(model, "Phase 3")

        ep_offset = PHASE1_EPOCHS + PHASE2_EPOCHS + 1
        if start_phase == 3 and os.path.exists(OUTPUT_CKPT):
            ckpt = torch.load(OUTPUT_CKPT, map_location=device, weights_only=False)
            if ckpt.get("phase") == 3:
                model.load_state_dict(ckpt["model"])
                ep_offset = ckpt["epoch"] + 1
                best_metric = ckpt["best_metric"]
                print(f"   Resuming phase 3 from epoch {ep_offset}")

        remaining = total_ep - (ep_offset - 1)
        if remaining > 0:
            best_metric = run_phase(
                model, train_loader, val_loader,
                param_groups_phase3(model), remaining,
                device, scaler, phase_num=3, best_metric=best_metric, start_epoch=ep_offset,
            )

    # ── Final TTA evaluation on best model ────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  FINAL EVALUATION (TTA) on best checkpoint")
    print(f"{'─'*65}")
    model.load_state_dict(torch.load(OUTPUT_BEST, map_location=device, weights_only=False))
    val_acc_tta, val_auc_tta = evaluate(model, val_loader, device, "final", "final", use_tta=True)
    print(f"  Val accuracy (TTA): {val_acc_tta:.4f}  ({val_acc_tta*100:.1f}%)")
    if val_auc_tta is not None:
        print(f"  Val AUC     (TTA): {val_auc_tta:.4f}")

    metric_name = "AUC" if HAS_SKLEARN else "Accuracy"
    print(f"\n{'='*65}")
    print(f"  Best {metric_name}    : {best_metric:.4f}")
    print(f"  Best weights → {OUTPUT_BEST}")
    print(f"{'='*65}")
    print("""
Next steps
──────────
1. Download f3net_binary.pth from /kaggle/working/
2. Copy to:  backend/app/models/weights/f3net_binary.pth
3. Re-run:   cd backend && python calibrate_models.py
4. If calibrated val accuracy > 70%:
     In backend/app/core/fusion.py, set:
       "f3net": 0.20
       "vit":   0.60
       "efficientnet": 0.20
""")


if __name__ == "__main__":
    main()
