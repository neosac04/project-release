# Deepfake Detector

A multi-model AI image forensics tool that detects deepfakes and AI-generated images using a 3-model ensemble — a Vision Transformer, a frequency-domain analyser, and a facial-texture CNN — with GradCAM heatmaps and plain-language explanations.

---

## Architecture

```
Frontend (React + Vite)   ──────────────────►  Backend (FastAPI)
   localhost:3000                                 localhost:8000
        │
        │  POST /api/v1/detect
        │  GET  /api/v1/results/{id}/heatmap/{model}
        ▼
  ┌─────────────────────────────────────┐
  │         3-Model Ensemble            │
  │                                     │
  │  ViT (weight 0.50)                  │
  │  HuggingFace dima806/deepfake_vs_   │
  │  real_image_detection               │
  │                                     │
  │  F3Net (weight 0.35)                │
  │  Frequency-domain binary classifier │
  │                                     │
  │  EfficientNet-B4 (weight 0.15)      │
  │  Face-texture binary classifier     │
  └─────────────────────────────────────┘
        │
        ▼
  Confidence-weighted fusion → final score + verdict
  GradCAM / Attention heatmaps per model
  Platt-scaled calibration per model
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.10 |
| Node.js | ≥ 18 |
| npm | ≥ 9 |

> **GPU optional.** The pipeline runs on CPU. A CUDA-capable GPU (4 GB+ VRAM) will speed up inference significantly.

---

## Project Structure

```
FinalYrProj/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI route handlers
│   │   ├── config/       # settings.py (paths, weights, thresholds)
│   │   ├── core/         # fusion.py, pipeline.py, preprocessing
│   │   ├── explainability/  # GradCAM implementation
│   │   └── models/       # ViT, F3Net, EfficientNet detectors
│   │       └── weights/  # ← place your .pth weight files here
│   ├── calibrate_models.py   # run once after training to calibrate
│   ├── train_effnet_head.py  # EfficientNet head fine-tuning script
│   └── train_f3net_head.py   # F3Net head fine-tuning script
├── frontend/
│   ├── src/
│   │   ├── api/          # axios client + detection API calls
│   │   ├── components/   # React UI components
│   │   ├── pages/        # UploadPage, ResultsPage
│   │   ├── store/        # Zustand state (detectionStore)
│   │   └── types/        # TypeScript types matching backend schema
│   ├── vite.config.ts    # Vite dev server (port 3000, /api proxy)
│   └── .env.local        # VITE_API_URL (auto-created on first run)
└── docker-compose.yml    # Optional Docker setup
```

---

## Quick Start

### 1 — Clone & set up

```bash
git clone https://github.com/neosac04/FinalYrProj.git
cd FinalYrProj
```

### 2 — Backend setup

```bash
cd backend

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3 — Add model weights

Place your trained `.pth` files in `backend/app/models/weights/`:

```
backend/app/models/weights/
├── efficientnet_binary.pth      # EfficientNet-B4 fine-tuned head
└── f3net_binary_best.pth        # F3Net fine-tuned binary head
```

> **ViT** is downloaded automatically from HuggingFace Hub on first run (`dima806/deepfake_vs_real_image_detection`). No manual download needed.

### 4 — Start the backend

```bash
# From the backend/ directory, with .venv active
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

You should see:
```
✅ EfficientNet loaded
✅ ViT loaded (fake_idx=1, label='Fake')
✅ F3Net loaded cleanly (0 missing, 0 unexpected)
INFO: Application startup complete.
```

Verify the API is running:
```bash
curl http://localhost:8000/api/v1/health
# → {"status":"ok"}
```

### 5 — Frontend setup

```bash
# From the project root
cd frontend
npm install
npm run dev
```

The app will be available at **http://localhost:3000**.

---

## Using the App

1. **Open** http://localhost:3000 in your browser
2. **Drag and drop** or **click** to upload an image (JPEG/PNG, max 10 MB)
3. Click **Analyse Image**
4. On the results page you will see:
   - **Verdict banner** — final fused confidence score (Real/Fake/Uncertain)
   - **Model vote table** — per-model predictions with individual confidence scores and fusion weights
   - **Heatmap viewer** — GradCAM/attention overlay showing suspicious regions (tabs: Ensemble / ViT / F3Net / EfficientNet). Use the opacity slider to blend with the original
   - **Explanation card** — plain-language summary of what each model found

---

## Environment Variables

### Backend

Create `backend/.env` (optional — defaults are shown):

```env
MODELS_DIR=./app/models/weights
EFFICIENTNET_WEIGHTS=efficientnet_binary.pth
F3NET_WEIGHTS=f3net_binary_best.pth
DEVICE=cpu           # set to "cuda" to use GPU
```

### Frontend

`frontend/.env.local` is created automatically, or you can create it manually:

```env
VITE_API_URL=/api/v1
```

The Vite dev server proxies all `/api` requests to `http://localhost:8000`.

---

## Docker (optional)

```bash
# From the project root
docker-compose up --build
```

| Service | Port |
|---------|------|
| Frontend | http://localhost:3000 |
| Backend | http://localhost:8000 |

---

## Model Details

| Model | Architecture | Task | Weights |
|-------|-------------|------|---------|
| **ViT** | `google/vit-base-patch16-224` fine-tuned on ~200k real/fake images | Full-image binary classification | Auto-downloaded from HuggingFace Hub |
| **F3Net** | ResNet backbone + frequency feature fusion (FAD + LFS) | Frequency-domain artefact detection | `f3net_binary_best.pth` |
| **EfficientNet-B4** | EfficientNet-B4 with fine-tuned binary head | Facial texture forensics | `efficientnet_binary.pth` |

### Fusion weights

| Model | Face detected | No face |
|-------|--------------|---------|
| ViT | 0.50 | 0.55 |
| F3Net | 0.35 | 0.40 |
| EfficientNet | 0.15 | 0.05 |

Verdict thresholds: **Fake** ≥ 0.62 · **Uncertain** 0.38–0.62 · **Real** ≤ 0.38

---

## Training / Calibration

To fine-tune the model heads on your own dataset:

```bash
# Fine-tune EfficientNet head
python backend/train_effnet_head.py

# Fine-tune F3Net head  
python backend/train_f3net_head.py

# Re-calibrate probability outputs after training
python backend/calibrate_models.py
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/detect` | Upload image, run detection. Returns full result JSON |
| `GET` | `/api/v1/results/{result_id}/heatmap/{model}` | PNG heatmap overlay. `model` ∈ `ensemble`, `vit`, `f3net`, `efficientnet` |

### Sample detection response

```json
{
  "result_id": "abc123",
  "verdict": "Fake",
  "final_score": 0.87,
  "is_uncertain": false,
  "face_detected": true,
  "total_inference_time_ms": 1240,
  "model_votes": {
    "vit":          { "fake_prob": 0.91, "real_prob": 0.09, "inference_time_ms": 450 },
    "f3net":        { "fake_prob": 0.83, "real_prob": 0.17, "inference_time_ms": 310 },
    "efficientnet": { "fake_prob": 0.76, "real_prob": 0.24, "inference_time_ms": 180 }
  },
  "fusion_weights": { "vit": 0.50, "f3net": 0.35, "efficientnet": 0.15 },
  "explanations": [
    "This image is likely a deepfake (87% confidence). All three detectors agree.",
    "ViT (global): Detected manipulation signatures across the full image (91% fake).",
    "F3Net (frequency): Found frequency-domain artefacts typical of GAN synthesis (83% fake).",
    "EfficientNet (texture): Facial texture shows signs of AI generation (76% fake)."
  ]
}
```
