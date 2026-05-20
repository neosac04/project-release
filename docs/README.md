# Deepfake Detector

Advanced multi-model deepfake and AI-generated image detection with explainability heatmaps,
anatomical analysis, frequency forensics, and camera fingerprint (PRNU) verification.

## Quick Start

### 1 — Download model weights
```bash
cd deepfake-detector
python models/download_weights.py
```

### 2 — Run with Docker Compose
```bash
docker compose up --build
```
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

### 3 — Or run locally (development)

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

| Model | Specialty | Source |
|---|---|---|
| **UnivFD** (CLIP ViT-L/14) | Universal — GAN, diffusion, all types | WisconsinAIVision/UniversalFakeDetect |
| **EfficientNet-B4** | Face forgeries (FF++ trained) | SCLBD/DeepfakeBench v1.0.1 |
| **Xception** | Texture artifacts (original FF++ baseline) | SCLBD/DeepfakeBench v1.0.1 |
| **DistilDIRE** | Diffusion-generated images | miraflow/DistilDIRE |

Ensemble weights shift dynamically based on the detected forgery type.

---

## Novel Features

- **Adaptive ensemble**: weights reallocate when diffusion/GAN/face-swap is suspected
- **Eye reflection symmetry**: specular highlights must mirror between eyes (fails in GANs)
- **Iris circularity**: GAN irises deviate from perfect circles (MediaPipe iris landmarks)
- **Jaw boundary seam detection**: Laplacian gradient discontinuity along face oval
- **PRNU camera fingerprint**: positive authenticity signal — real cameras leave a unique noise signature
- **Model disagreement indicator**: high variance between models surfaces as first-class uncertainty
- **PCA feature space plot**: shows where your image falls among known fake/real clusters

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/detect` | Upload image, run full pipeline |
| GET | `/api/v1/heatmap/{result_id}/{model}` | PNG heatmap overlay (ensemble/univfd/efficientnet/freqnet) |
| GET | `/api/v1/result/{result_id}` | Retrieve cached result |
| GET | `/api/v1/models/status` | Which models are loaded |
| GET | `/api/v1/health` | Liveness probe |

---

## Without GPU

All models run on CPU. Inference takes ~5-15 seconds depending on hardware.
Set `DEVICE=cuda` in `.env` if a GPU is available for ~5× speedup.

---

## Project Structure

```
deepfake-detector/
├── models/               ← pretrained weights (download_weights.py)
├── backend/
│   └── app/
│       ├── models/       ← UnivFD, EfficientNet, FreqNet, DIRE wrappers
│       ├── analysis/     ← facial, frequency, PRNU, color, compression
│       ├── core/         ← pipeline, ensemble, fake_type_classifier
│       ├── explainability/← GradCAM++, attention rollout, overlay
│       └── api/          ← FastAPI endpoints
└── frontend/
    └── src/
        ├── components/   ← all UI panels
        ├── pages/        ← UploadPage, ResultsPage
        └── store/        ← Zustand state
```
# jazz
