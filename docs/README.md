# Deepfake Detector

Multi-model deepfake and AI-generated image/video detection with explainability heatmaps,
facial landmark analysis, frequency forensics, and skin quality scoring.

## Quick Start

### Run with Docker Compose
```bash
cd release
docker compose up --build
```
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

### Run locally (development)

**Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend**
```bash
cd frontend
npm install
npm run dev       # runs on http://localhost:3000
```

---

## Models

| Model | Specialty | Source | Weight |
|---|---|---|---|
| **ViT** (dima806) | Full-image synthesis detection — AUC 0.999 | HuggingFace hub (auto-download) | 30% |
| **SigLIP** (fine-tuned) | Vision-language face classification — 94.44% accuracy | `app/models/weights/siglip/` | 25% |
| **F3Net** | Frequency-domain DCT deepfake detection — AUC 0.958 | `app/models/weights/f3net_binary_best.pth` | 20% |
| **EfficientNet-B4** | Facial texture micro-artifact CNN — AUC 0.764 | `app/models/weights/efficientnet_binary.pth` | 10% |
| **Hive AI API** | External AI-image detection oracle | API key via `EXT_API_KEY` | 15% |
| **XceptionNet** *(optional)* | Localised manipulation artifacts | `app/models/weights/xception_best.pth` | — |

Models that have no weight file are silently skipped at startup. ViT auto-downloads on first run.
Fusion weights are re-normalised across loaded models only.

---

## Hive API Setup

The Hive AI detector is enabled when `EXT_API_KEY` is set. Add to `backend/.env`:

```bash
EXT_API_URL=https://api.thehive.ai/api/v2/task/sync
EXT_API_KEY=your_hive_api_key
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/detect` | Upload image, run full pipeline |
| POST | `/api/v1/detect/video` | Upload video, run frame-based detection |
| GET | `/api/v1/heatmap/{result_id}/{model}` | PNG heatmap overlay (`ensemble`/`vit`/`siglip`/`f3net`/`efficientnet`) |
| GET | `/api/v1/result/{result_id}` | Retrieve cached image result |
| GET | `/api/v1/video/result/{result_id}` | Retrieve cached video result |
| GET | `/api/v1/models/status` | Which models are loaded |
| GET | `/api/v1/health` | Liveness probe |
| GET | `/api/v1/ready` | Readiness probe (reports model + asset status) |

---

## Without GPU

All models run on CPU. Inference takes ~5–20 seconds depending on hardware.
Set `DEVICE=cuda` in `.env` if a GPU is available.

---

## Project Structure

```
release/
├── backend/
│   └── app/
│       ├── models/weights/   ← .pth weight files + siglip/ directory
│       ├── models/mediapipe/ ← face_landmarker.task
│       ├── models/           ← detector classes (efficientnet, vit, siglip, f3net, hive)
│       ├── analysis/         ← FFT, texture, face symmetry, skin quality
│       ├── core/             ← pipeline, fusion engine
│       ├── explainability/   ← GradCAM, overlay
│       ├── video/            ← frame extractor, temporal aggregator
│       └── api/endpoints/    ← detect, video_detection, visualization, health
└── frontend/
    └── src/
        ├── components/       ← VerdictBanner, ModelVoteTable, HeatmapViewer, FrameTimeline
        ├── pages/            ← UploadPage, ResultsPage
        └── store/            ← Zustand detection state
```
