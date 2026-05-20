# Project Description

This project implements both image and video deepfake detection in a single FastAPI-based system.

The image pipeline runs a multi-model ensemble over an uploaded image, while the video pipeline samples frames from an uploaded video, reuses the image pipeline on each frame, and aggregates the per-frame scores into a video-level verdict.

## What The Project Does

- Detects whether an uploaded image is real or fake.
- Detects whether an uploaded video contains deepfake content by analysing sampled frames.
- Returns a final score, verdict, per-model votes, and human-readable explanations.
- Produces analysis metrics for explainability, including frequency, texture, symmetry, and skin-related signals.
- Stores and serves cached detection results for later retrieval.

## Implemented Detection Modes

### Image Detection

The image endpoint is implemented in the backend detection pipeline and exposed through the API at `/api/v1/detect`.

The image pipeline:

- Loads all available detectors from the model registry.
- Detects the largest face in the image when possible.
- Uses face-cropped input for face-focused models and full-image input for global models.
- Runs all loaded detectors in parallel.
- Fuses the model outputs into one final score and verdict.
- Generates explanation text and analysis metrics for the response.

### Video Detection

The video endpoint is implemented at `/api/v1/detect/video`.

The video pipeline:

- Extracts evenly spaced frames from the uploaded video.
- Runs the same image deepfake pipeline on each frame.
- Aggregates per-frame scores using a confident temporal strategy.
- Reports frame-level results, a final video verdict, temporal consistency, and summary explanations.

## Models Used

The registry in `backend/app/models/registry.py` loads the following detectors when their weights or runtime credentials are available:

| Model | Role | Input style |
|---|---|---|
| EfficientNet | Facial texture / local artifact detection | Face crop |
| ViT | Global image structure and synthesis patterns | Full image |
| F3Net | Frequency-domain deepfake detection | Full image |
| SigLIP | Vision-language based fake detection | Face crop |
| XceptionNet | Optional manipulation-artifact detector | Face crop |
| Hive | Optional external AI-generated image detector | Face crop |

### Input Routing

The pipeline routes models to the input view they were trained for:

- Face-cropped models: EfficientNet, XceptionNet, SigLIP
- Full-image models: F3Net, ViT

If no face is detected, the face-cropped models fall back to the full image.

## Core Functions And Features

- `backend/app/core/pipeline.py`: main image detection workflow, model fusion, and explanation generation.
- `backend/app/video/pipeline.py`: video detection workflow built on top of the image pipeline.
- `backend/app/video/frame_extractor.py`: extracts evenly spaced frames from video bytes.
- `backend/app/video/temporal_aggregator.py`: combines frame scores into a final video score.
- `backend/app/analysis/*`: computes supporting forensic signals such as FFT, texture, face symmetry, and skin metrics.
- `backend/app/explainability/*`: prepares heatmap and overlay outputs for model explainability.
- `backend/app/api/endpoints/detection.py`: image detection and cached result retrieval.
- `backend/app/api/endpoints/video_detection.py`: video detection and cached video result retrieval.

## API Summary

- `POST /api/v1/detect` — upload an image and run deepfake detection.
- `GET /api/v1/result/{result_id}` — fetch a cached image result.
- `POST /api/v1/detect/video` — upload a video and run frame-based deepfake detection.
- `GET /api/v1/video/result/{result_id}` — fetch a cached video result.
- `GET /api/v1/models/status` — view which models are loaded.
- `GET /api/v1/health` — health check.

## Bottom Line

Yes, both image and video deepfake detection are implemented. The video path is not a separate unrelated model; it is a frame-based video pipeline that reuses the image ensemble and adds temporal aggregation on top.