from __future__ import annotations
import io
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_clip_transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(CLIP_MEAN, CLIP_STD),
])

_imagenet_transform = T.Compose([
    T.Resize(380, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(380),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_freq_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
])

# DeepfakeBench-style: 256×256, normalize to [-1, 1]
_dfb_transform = T.Compose([
    T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def preprocess(image: Image.Image) -> dict:
    rgb = image.convert("RGB")
    raw_np = np.array(rgb)
    return {
        "clip_tensor": _clip_transform(rgb),
        "imagenet_tensor": _imagenet_transform(rgb),
        "dfb_tensor": _dfb_transform(rgb),
        "freq_pil": _freq_transform(rgb),
        "raw_np": raw_np,
        "gray_np": np.array(rgb.convert("L")),
        "pil": rgb,
    }


def bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()
