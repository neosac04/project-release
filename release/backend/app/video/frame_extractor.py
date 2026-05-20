"""
Extract evenly-spaced frames from a video file using OpenCV.
Adapted from dfdc_deepfake_challenge/kernel_utils.py VideoReader class.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
from PIL import Image


def extract_frames(
    video_bytes: bytes,
    num_frames: int = 32,
) -> Tuple[List[Image.Image], float, List[int]]:
    """
    Extract num_frames evenly-spaced frames from video bytes.

    Returns:
        frames:        List of PIL Images in RGB colour space.
        fps:           Frames-per-second of the source video.
        frame_indices: Actual frame indices extracted.
    """
    suffix = ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            cap.release()
            with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp2:
                tmp2.write(video_bytes)
                tmp_path2 = tmp2.name
            cap = cv2.VideoCapture(tmp_path2)
            if not cap.isOpened():
                raise ValueError(
                    "Could not open video file. Ensure the file is a valid MP4, WebM, AVI, or MOV."
                )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        if total_frames <= 0:
            cap.release()
            raise ValueError("Video appears to have no readable frames.")

        n = min(num_frames, total_frames)
        if n <= 1:
            indices = [0]
        elif n == total_frames:
            indices = list(range(total_frames))
        else:
            indices = [int(round(i * (total_frames - 1) / (n - 1))) for i in range(n)]

        frames: List[Image.Image] = []
        read_indices: List[int] = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
                read_indices.append(idx)

        cap.release()

        if not frames:
            raise ValueError("No frames could be decoded from the video.")

        return frames, fps, read_indices

    finally:
        Path(tmp_path).unlink(missing_ok=True)
