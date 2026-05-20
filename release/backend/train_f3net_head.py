"""
Head-only fine-tuning of F3Net's classifier on the Dataset/ folder.

Strategy mirrors train_effnet_head.py:
  - Load the pre-trained f3net_best.pth (Xception backbone + FAD head)
  - Freeze EVERYTHING except `backbone.last_linear`
  - Train just the 2-class linear head on Dataset/Train, validate on Dataset/Validation
  - Save best-by-val-accuracy checkpoint to f3net_binary.pth

Class order from torchvision ImageFolder on Dataset/: {Fake: 0, Real: 1}.
After training, F3Net's `probs[0]` will be fake_prob (matches the existing
F3Net detector class).

Run:
  cd backend
  python train_f3net_head.py \
    --dataset-root /Users/user/Downloads/Dataset \
    --epochs 10 --batch-size 32 --lr 1e-3 \
    --output ./app/models/weights/f3net_binary.pth

Notes:
  - Backbone + FAD head stay frozen, so training is cheap (CPU-feasible).
  - The FAD head's `learnable` filters technically have requires_grad=True in
    the saved checkpoint, but we freeze them too so we don't drift the feature
    extractor.
  - After training, re-run calibrate_models.py and re-evaluate the pipeline;
    bump F3Net's fusion weight (app/core/fusion.py) once it shows real signal.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.models.f3net import _F3NetWrapper


def build_transforms() -> transforms.Compose:
    # Must match the inference-time `dfb_tensor` preprocessing exactly:
    # 256×256 with mean/std = 0.5 (so input lands in [-1, 1]).
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_model(pretrained_path: Path, device: torch.device) -> nn.Module:
    model = _F3NetWrapper()
    state = torch.load(str(pretrained_path), map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"⚠️ load — missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("✅ F3Net base weights loaded (0 missing, 0 unexpected)")

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Re-init the head with fresh random weights so we train from scratch
    head = model.backbone.last_linear
    if isinstance(head, nn.Sequential):
        # last_linear = Sequential(Dropout, Linear(2048, 2))
        new_linear = nn.Linear(2048, 2)
        head[1] = new_linear
        for p in head[1].parameters():
            p.requires_grad = True
    else:
        new_linear = nn.Linear(2048, 2)
        model.backbone.last_linear = new_linear
        for p in model.backbone.last_linear.parameters():
            p.requires_grad = True

    return model.to(device)


def trainable_params(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += images.size(0)
    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += images.size(0)
    return total_correct / max(total_samples, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Head-only F3Net fine-tuning for binary deepfake detection.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Folder containing Train/ and Validation/.")
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=Path(__file__).resolve().parent / "app" / "models" / "weights" / "f3net_best.pth",
        help="Path to f3net_best.pth (Xception+FAD backbone init).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "app" / "models" / "weights" / "f3net_binary.pth",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader workers (set 0 on macOS if you hit fork issues)."
    )
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    transform = build_transforms()

    train_dir = args.dataset_root / "Train"
    val_dir = args.dataset_root / "Validation"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected layout:\n  {args.dataset_root}/Train/Fake, {args.dataset_root}/Train/Real,\n"
            f"  {args.dataset_root}/Validation/Fake, {args.dataset_root}/Validation/Real"
        )

    train_ds = datasets.ImageFolder(root=str(train_dir), transform=transform)
    val_ds = datasets.ImageFolder(root=str(val_dir), transform=transform)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print("Class mapping:", train_ds.class_to_idx)  # expect {'Fake': 0, 'Real': 1}
    if train_ds.class_to_idx.get("Fake") != 0:
        print("⚠️ WARNING: Fake is not class 0 — the F3Net detector assumes Fake=0.")

    model = build_model(args.pretrained, device)
    trainable = trainable_params(model)
    n_trainable = sum(p.numel() for p in trainable)
    print(f"Trainable params: {n_trainable:,}  (head only)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    best_val_acc = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss: {train_loss:.4f} | train acc: {train_acc:.4f} | val acc: {val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  ✓ Saved best checkpoint → {args.output}")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"\nNext steps:")
    print(f"  1. Update settings.py: set F3NET_MODEL_PATH to {args.output.name}")
    print(f"  2. Re-run calibrate_models.py to refresh F3Net's Platt scaling")
    print(f"  3. In app/core/fusion.py, raise F3Net's weight (e.g. 0.20).")


if __name__ == "__main__":
    main()
