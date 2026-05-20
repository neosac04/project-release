#!/usr/bin/env python3
"""
Download pretrained model weights for the Deepfake Detector pipeline.

Run from the repo root:
    python models/download_weights.py

What gets downloaded:
    MediaPipe FaceLandmarker  → models/mediapipe/face_landmarker.task   (~29 MB)
    UnivFD linear probe       → models/univfd.pth                        (~3 KB)
    EfficientNet-B4 (FF++)    → models/efficientnet.pth                  (~75 MB)
    Xception    (FF++)        → models/xception.pth                      (~85 MB)

Sources:
    MediaPipe  : storage.googleapis.com  (stable Google CDN)
    UnivFD     : huggingface.co/WisconsinAIVision/UniversalFakeDetect
    EfficientNet / Xception : huggingface.co/SCLBD/DeepfakeBench
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import textwrap
import urllib.request
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class ModelSpec(NamedTuple):
    name: str
    dest: str          # path relative to repo root
    url: str
    sha256: str | None # None = skip checksum (large files checked by size only)
    min_bytes: int     # sanity-check minimum size


MODELS: list[ModelSpec] = [
    ModelSpec(
        name="MediaPipe FaceLandmarker",
        dest="models/mediapipe/face_landmarker.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models"
            "/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        ),
        sha256=None,
        min_bytes=20 * 1024 * 1024,   # ~29 MB on disk
    ),
    ModelSpec(
        name="UnivFD linear probe (CLIP ViT-L/14)",
        dest="models/univfd.pth",
        url=(
            "https://huggingface.co/WisconsinAIVision/UniversalFakeDetect"
            "/resolve/main/fc_weights.pth"
        ),
        sha256=None,
        min_bytes=2 * 1024,   # tiny linear layer ~3 KB
    ),
    ModelSpec(
        name="EfficientNet-B4 (DeepfakeBench / FF++)",
        dest="models/efficientnet.pth",
        url=(
            "https://huggingface.co/SCLBD/DeepfakeBench"
            "/resolve/main/pretrained/EfficientNetB4/best.pth"
        ),
        sha256=None,
        min_bytes=50 * 1024 * 1024,   # ~75 MB
    ),
    ModelSpec(
        name="Xception (DeepfakeBench / FF++)",
        dest="models/xception.pth",
        url=(
            "https://huggingface.co/SCLBD/DeepfakeBench"
            "/resolve/main/pretrained/Xception/best.pth"
        ),
        sha256=None,
        min_bytes=50 * 1024 * 1024,   # ~85 MB
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
CYAN = "\033[36m"


def _c(color: str, text: str) -> str:
    """Wrap text in ANSI colour if stdout is a tty."""
    if sys.stdout.isatty():
        return f"{color}{text}{RESET}"
    return text


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class _ProgressReporter:
    """urllib reporthook that prints a live progress bar."""

    def __init__(self, name: str) -> None:
        self._name = name[:40]
        self._last_pct = -1

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size <= 0:
            # Server didn't send Content-Length — just print bytes
            print(f"\r  {self._name}: {_fmt_bytes(downloaded)}   ", end="", flush=True)
            return
        pct = min(100, int(downloaded * 100 / total_size))
        if pct == self._last_pct:
            return
        self._last_pct = pct
        bar_len = 30
        filled = bar_len * pct // 100
        bar = "█" * filled + "░" * (bar_len - filled)
        line = f"\r  [{bar}] {pct:3d}%  {_fmt_bytes(downloaded)} / {_fmt_bytes(total_size)}  "
        print(line, end="", flush=True)


# ---------------------------------------------------------------------------
# Core download logic
# ---------------------------------------------------------------------------

def _already_valid(dest: Path, spec: ModelSpec) -> bool:
    """Return True if the file exists and passes size/checksum checks."""
    if not dest.exists():
        return False
    size = dest.stat().st_size
    if size < spec.min_bytes:
        print(
            _c(YELLOW, f"  ⚠  {dest.name} exists but looks truncated "
               f"({_fmt_bytes(size)} < {_fmt_bytes(spec.min_bytes)}). Re-downloading.")
        )
        return False
    if spec.sha256:
        actual = _sha256_file(dest)
        if actual != spec.sha256:
            print(_c(YELLOW, f"  ⚠  {dest.name} checksum mismatch. Re-downloading."))
            return False
    print(_c(GREEN, f"  ✓  {spec.name} already present ({_fmt_bytes(size)})"))
    return True


def _download_one(spec: ModelSpec, repo_root: Path, force: bool) -> bool:
    """Download a single model. Returns True on success."""
    dest = repo_root / spec.dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{_c(BOLD, spec.name)}")
    print(f"  → {spec.dest}")

    if not force and _already_valid(dest, spec):
        return True

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        reporter = _ProgressReporter(spec.name)
        urllib.request.urlretrieve(spec.url, tmp, reporthook=reporter)
        print()  # newline after progress bar
    except Exception as exc:
        print()
        tmp.unlink(missing_ok=True)
        print(_c(RED, f"  ✗  Download failed: {exc}"))
        return False

    # Validate size
    size = tmp.stat().st_size
    if size < spec.min_bytes:
        tmp.unlink(missing_ok=True)
        print(_c(RED, f"  ✗  File too small ({_fmt_bytes(size)}). URL may be wrong or rate-limited."))
        return False

    # Validate checksum
    if spec.sha256:
        actual = _sha256_file(tmp)
        if actual != spec.sha256:
            tmp.unlink(missing_ok=True)
            print(_c(RED, f"  ✗  SHA-256 mismatch (got {actual[:16]}…). File is corrupt."))
            return False

    tmp.rename(dest)
    print(_c(GREEN, f"  ✓  Saved → {dest}  ({_fmt_bytes(size)})"))
    return True


# ---------------------------------------------------------------------------
# Manual-download instructions (fallback)
# ---------------------------------------------------------------------------

MANUAL_INSTRUCTIONS = {
    "univfd.pth": textwrap.dedent("""\
        UnivFD weights (CLIP linear probe):
          1. Visit https://github.com/WisconsinAIVision/UniversalFakeDetect
          2. Follow the download link in their README for 'fc_weights.pth'
          3. Save it to  models/univfd.pth
    """),
    "efficientnet.pth": textwrap.dedent("""\
        EfficientNet-B4 weights (DeepfakeBench):
          1. Visit https://github.com/SCLBD/DeepfakeBench
          2. Download the EfficientNetB4 checkpoint from their model zoo
          3. Save it to  models/efficientnet.pth
    """),
    "xception.pth": textwrap.dedent("""\
        Xception weights (DeepfakeBench):
          1. Visit https://github.com/SCLBD/DeepfakeBench
          2. Download the Xception checkpoint from their model zoo
          3. Save it to  models/xception.pth
    """),
    "face_landmarker.task": textwrap.dedent("""\
        MediaPipe FaceLandmarker:
          Direct URL (paste in browser or wget):
          https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
          Save to  models/mediapipe/face_landmarker.task
    """),
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download model weights for the Deepfake Detector."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if the file already exists.",
    )
    parser.add_argument(
        "--only", metavar="NAME",
        help="Download only the model whose dest filename matches NAME "
             "(e.g. univfd.pth, face_landmarker.task).",
    )
    parser.add_argument(
        "--root", metavar="DIR", default=None,
        help="Repo root directory. Defaults to the parent of the 'models/' folder "
             "containing this script.",
    )
    args = parser.parse_args()

    # Resolve repo root
    script_dir = Path(__file__).resolve().parent   # models/
    repo_root = Path(args.root).resolve() if args.root else script_dir.parent

    print(_c(BOLD + CYAN, "\n═══ Deepfake Detector — Model Weight Downloader ═══\n"))
    print(f"  Repo root : {repo_root}")
    print(f"  Models dir: {repo_root / 'models'}\n")

    # Filter models if --only was given
    specs = MODELS
    if args.only:
        specs = [s for s in MODELS if args.only in s.dest]
        if not specs:
            print(_c(RED, f"No model matches '{args.only}'. Available:"))
            for s in MODELS:
                print(f"  {s.dest}")
            return 1

    results: dict[str, bool] = {}
    for spec in specs:
        ok = _download_one(spec, repo_root, force=args.force)
        results[spec.dest] = ok

    # Summary
    print(f"\n{_c(BOLD, '─── Summary ─────────────────────────────────────')}")
    failed: list[ModelSpec] = []
    for spec in specs:
        ok = results[spec.dest]
        mark = _c(GREEN, "✓") if ok else _c(RED, "✗")
        print(f"  {mark}  {spec.name:<45} {spec.dest}")
        if not ok:
            failed.append(spec)

    if failed:
        print(_c(YELLOW, "\n⚠  Some downloads failed. Manual download instructions:\n"))
        for spec in failed:
            fname = Path(spec.dest).name
            instructions = MANUAL_INSTRUCTIONS.get(fname)
            if instructions:
                print(textwrap.indent(instructions, "  "))
        print(
            _c(YELLOW,
               "  Once downloaded, re-run this script to validate the files.\n")
        )
        return 1

    print(_c(GREEN, "\n✓  All models ready. You can now start the backend.\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
