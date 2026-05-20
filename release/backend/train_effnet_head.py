from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4


def build_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_model(device: torch.device, local_backbone: Path | None = None) -> nn.Module:
    """Build EfficientNet-B4 with a 2-class head.

    Loads ImageNet backbone from *local_backbone* when provided (avoids SSL
    issues on Python 3.14 macOS).  Falls back to torchvision download if the
    path is not given or doesn't exist.
    """
    if local_backbone and local_backbone.exists():
        # Build architecture without pretrained weights, then load locally.
        model = efficientnet_b4(weights=None)
        state = torch.load(str(local_backbone), map_location="cpu", weights_only=False)
        # The local file keeps a 1 000-class head — strict=False is intentional;
        # we replace classifier[1] immediately below.
        missing, unexpected = model.load_state_dict(state, strict=False)
        # Only the classifier head keys should be missing/unexpected — warn if more.
        non_head = [k for k in (missing + unexpected) if "classifier" not in k]
        if non_head:
            print(f"WARNING: unexpected non-head keys: {non_head[:5]} …")
        print(f"Loaded backbone from local file: {local_backbone}")
    else:
        print("Local backbone not found — downloading from PyTorch CDN …")
        model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

    model.classifier[1] = nn.Linear(1792, 2)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    return model.to(device)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


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


def collect_images(paths: list[Path]) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files: list[Path] = []
    for root in paths:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
    return sorted(files)


async def run_pipeline_inference(image_paths: list[Path], effnet_path: Path) -> None:
    os.environ["EFFICIENTNET_MODEL_PATH"] = str(effnet_path)

    from app.core.pipeline import DetectionPipeline
    from app.models.registry import ModelRegistry
    from app.config.settings import settings

    registry = ModelRegistry.get_instance()
    registry.load_all(settings.models_dir, settings.device)
    pipeline = DetectionPipeline()

    for path in image_paths:
        image = Image.open(path).convert("RGB")
        result = await pipeline.run(image)
        print(f"{path}")
        print("univfd_score:", result.univfd_score)
        print("efficientnet_score:", result.efficientnet_score)
        print("final_score:", result.final_score)
        print("verdict:", result.verdict)
        print("-" * 40)


def main() -> None:
    parser = argparse.ArgumentParser(description="Head-only EfficientNet-B4 fine-tuning for binary deepfake detection.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"), help="Root containing train/ and val/ folders.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "app" / "models" / "weights" / "efficientnet_binary.pth",
        help="Checkpoint output path.",
    )
    parser.add_argument(
        "--backbone-weights",
        type=Path,
        default=Path(__file__).resolve().parent / "app" / "models" / "weights" / "efficientnet_b4.pth",
        help="Local ImageNet backbone .pth to load instead of downloading (avoids SSL issues).",
    )
    parser.add_argument(
        "--eval-dirs",
        nargs="*",
        default=[],
        help="Optional dirs for post-train pipeline inference checks (e.g. FF++, Celeb-DF, diffusion, web-real).",
    )
    parser.add_argument("--max-eval-images", type=int, default=24)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    transform = build_transforms()

    # Support both lower-case (train/val) and Title-Case (Train/Validation) folder names
    def _find_dir(root: Path, *candidates: str) -> Path:
        for name in candidates:
            p = root / name
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Could not find any of {candidates} inside {root}. "
            f"Directory contents: {[d.name for d in root.iterdir() if d.is_dir()]}"
        )

    train_dir = _find_dir(args.dataset_root, "train", "Train")
    val_dir   = _find_dir(args.dataset_root, "val", "Val", "Validation", "validation")
    print(f"Train dir : {train_dir}")
    print(f"Val   dir : {val_dir}")

    train_ds = datasets.ImageFolder(root=str(train_dir), transform=transform)
    val_ds = datasets.ImageFolder(root=str(val_dir), transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("Class mapping:", train_ds.class_to_idx)
    model = build_model(device, local_backbone=args.backbone_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)

    best_val_acc = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss: {train_loss:.4f} | train accuracy: {train_acc:.4f} | val accuracy: {val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"Saved best checkpoint to: {args.output}")

    print(f"Best validation accuracy: {best_val_acc:.4f}")

    eval_paths = [Path(p) for p in args.eval_dirs]
    if eval_paths:
        image_paths = collect_images(eval_paths)[: args.max_eval_images]
        print(f"Running pipeline inference on {len(image_paths)} image(s) using {args.output}")
        asyncio.run(run_pipeline_inference(image_paths, args.output))


if __name__ == "__main__":
    main()
