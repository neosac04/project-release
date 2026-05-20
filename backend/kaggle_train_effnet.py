"""
EfficientNet-B4 Head-Only Training — Kaggle GPU Notebook
=========================================================
Instructions:
  1. Create a new Kaggle notebook (Notebook → + New Notebook)
  2. Add the dataset: Data → Add Data → search "140k real and fake faces" by xhlulu
  3. Enable GPU: Settings → Accelerator → GPU T4 x1
  4. Paste the entire contents of this file into a single code cell and Run All
  5. After training, the output weights file is in /kaggle/working/efficientnet_binary.pth
     Download it from the Kaggle output panel (right sidebar → Output)
  6. Copy the downloaded file to:
       FinalYrProj/backend/app/models/weights/efficientnet_binary.pth

Expected results:
  - Each epoch: ~8–12 min on T4
  - 10 epochs total: ~2 hrs
  - Target val accuracy: 70–80%

Dataset structure Kaggle provides:
  /kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/
    train/fake/  train/real/
    valid/fake/  valid/real/
    test/fake/   test/real/
"""

# ── 0. Imports ──────────────────────────────────────────────────────────────
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

# ── 1. Config ────────────────────────────────────────────────────────────────
OUTPUT_PATH      = Path("/kaggle/working/efficientnet_binary.pth")
CHECKPOINT_DIR   = Path("/kaggle/working/checkpoints")

# 5 epochs is enough — previous run hit 0.824 val acc by ~ep5
EPOCHS     = 5
BATCH_SIZE = 128      # AMP lets 128 fit in T4's 16 GB at 380×380
LR         = 1e-3
NUM_WORKERS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
assert device.type == "cuda", "GPU not active — enable it in Notebook Settings → Accelerator"

# ── 2. Discover dataset paths ─────────────────────────────────────────────────
# Mount point confirmed: /kaggle/input/datasets/mathstudent04/140k-real-and-fake-faces
# Internal structure unknown — scan up to 2 levels inside to find train/val.
DATASET_MOUNT = Path("/kaggle/input/datasets/mathstudent04/140k-real-and-fake-faces")

TRAIN_NAMES = ("train", "Train")
VAL_NAMES   = ("valid", "Valid", "val", "Val", "Validation", "validation", "test", "Test")

def _find_split(root: Path, *names: str) -> Path | None:
    """Check root directly, then one level of subdirs."""
    for name in names:
        p = root / name
        if p.exists():
            return p
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        for name in names:
            p = sub / name
            if p.exists():
                return p
    return None

TRAIN_DIR = _find_split(DATASET_MOUNT, *TRAIN_NAMES)
VAL_DIR   = _find_split(DATASET_MOUNT, *VAL_NAMES)

assert TRAIN_DIR, (
    f"Could not find a train subfolder inside {DATASET_MOUNT}.\n"
    f"Contents: {[p.name for p in DATASET_MOUNT.iterdir()]}"
)
assert VAL_DIR, (
    f"Found train at {TRAIN_DIR} but no val/valid folder.\n"
    f"Parent contents: {[p.name for p in TRAIN_DIR.parent.iterdir()]}"
)

print(f"Train dir : {TRAIN_DIR}")
print(f"Val   dir : {VAL_DIR}")

# ── 3. Transforms ────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── 4. Datasets & loaders ────────────────────────────────────────────────────
train_ds = datasets.ImageFolder(root=str(TRAIN_DIR), transform=transform)
val_ds   = datasets.ImageFolder(root=str(VAL_DIR),   transform=transform)

print(f"Class mapping : {train_ds.class_to_idx}")
print(f"Train samples : {len(train_ds):,}")
print(f"Val   samples : {len(val_ds):,}")

# IMPORTANT: confirm Fake=0, Real=1 — the inference code depends on this
fake_idx = train_ds.class_to_idx.get("fake", train_ds.class_to_idx.get("Fake"))
assert fake_idx == 0, (
    f"Unexpected class order: {train_ds.class_to_idx}\n"
    f"Fake must be index 0 (got {fake_idx}). "
    "If fake=1, stop here and report back — the inference code must be adjusted."
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

# ── 5. Model (head-only) ─────────────────────────────────────────────────────
model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(1792, 2)

# Freeze backbone, train head only
for p in model.parameters():
    p.requires_grad = False
for p in model.classifier.parameters():
    p.requires_grad = True

model = model.to(device)
print(f"\nTrainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Frozen   params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")

# ── 6. Loss & optimiser ──────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)
# Optional LR scheduler — reduces LR by 0.5 if val acc plateaus for 2 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)

# ── 7. Training loop ─────────────────────────────────────────────────────────
best_val_acc = 0.0
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

import sys
best_val_acc = 0.0
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# AMP scaler — ~1.8x speedup on T4, no accuracy change
scaler = torch.cuda.amp.GradScaler()

for epoch in range(1, EPOCHS + 1):
    # — Train —
    model.train()
    total_loss = total_correct = total_samples = 0
    for batch_idx, (images, labels) in enumerate(train_loader, 1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():       # mixed precision forward pass
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss    += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += images.size(0)

        if batch_idx % 100 == 0 or batch_idx == len(train_loader):
            print(f"  Ep {epoch}/{EPOCHS} | batch {batch_idx}/{len(train_loader)} "
                  f"| loss: {total_loss/total_samples:.4f} | acc: {total_correct/total_samples:.4f}",
                  flush=True)

    train_loss = total_loss / total_samples
    train_acc  = total_correct / total_samples

    # — Validate —
    model.eval()
    val_correct = val_total = 0
    with torch.no_grad(), torch.cuda.amp.autocast():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            val_correct += (logits.argmax(1) == labels).sum().item()
            val_total   += images.size(0)
    val_acc = val_correct / val_total

    scheduler.step(val_acc)

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"loss: {train_loss:.4f} | train acc: {train_acc:.4f} | val acc: {val_acc:.4f}")

    # Save per-epoch checkpoint — if session resets you lose at most 1 epoch
    epoch_ckpt = CHECKPOINT_DIR / f"ep{epoch:02d}_val{val_acc:.4f}.pth"
    torch.save(model.state_dict(), epoch_ckpt)
    print(f"  Checkpoint saved: {epoch_ckpt.name}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), OUTPUT_PATH)
        print(f"  ✓ New best saved  (val acc: {best_val_acc:.4f})")

print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
print(f"Best weights      : {OUTPUT_PATH}")
print(f"All checkpoints   : {CHECKPOINT_DIR}")

# ── 8. Quick sanity check on saved weights ───────────────────────────────────
ck = torch.load(str(OUTPUT_PATH), map_location="cpu", weights_only=False)
head_shape = tuple(ck["classifier.1.weight"].shape)
print(f"\nSaved head shape: {head_shape}")
assert head_shape == (2, 1792), f"Unexpected shape {head_shape} — expected (2, 1792)"
print("✓ Head shape correct: (2, 1792)")
print("\nDownload efficientnet_binary.pth from the Kaggle Output panel →")
print("then copy to:  FinalYrProj/backend/app/models/weights/efficientnet_binary.pth")

print("then copy it to:  FinalYrProj/backend/app/models/weights/efficientnet_binary.pth")
