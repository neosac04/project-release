# Project Issues Audit

Date: 2026-05-05

Scope: Static review of the local `jazz` project, focused on prediction failures, backend connectivity problems, frontend/API integration, model loading, visualization, and repository health.

No source files were changed as part of this audit.

## Executive Summary

The project has several high-impact problems that can explain both unreliable predictions and intermittent backend failures:

- Model assets are missing from the repository, and the documented download script does not exist.
- The first prediction request can crash when the MediaPipe face landmarker model is absent.
- Backend health checks report `ok` even when no ML models are loaded.
- Face crops likely suffer from RGB/BGR channel confusion, which can damage model inputs.
- Missing or incompatible weights are handled too silently, causing neutral or misleading predictions.
- Heatmaps and landmarks are generated on cropped images but displayed over the full original image, so visual explanations can be misaligned.

## Critical Runtime Issues

### Missing Model Assets

There is no `models/` directory, no `.pth` model weights, and no MediaPipe `.task` file in the repository.

The README tells users to run:

```bash
python models/download_weights.py
```

but no such script exists.

Relevant files:

- `README.md`
- `backend/app/config/settings.py`
- `.gitignore`

Impact:

- The backend may start with zero learned models loaded.
- Predictions may be based on fallback or heuristic behavior only.
- Users receive results that look valid but are not backed by the advertised models.

### MediaPipe Model Can Crash First Detection

`DetectionPipeline()` constructs `FacialAnalyzer` during the first `/detect` request. `FacialAnalyzer` raises `FileNotFoundError` if this file is absent:

```text
models/mediapipe/face_landmarker.task
```

Relevant files:

- `backend/app/core/pipeline.py`
- `backend/app/preprocessing/facial.py`

Impact:

- Backend may appear reachable.
- First image submission can fail with a server error.
- This matches the symptom of the frontend sometimes not connecting or failing after upload.

### Health Endpoint Is Misleading

`/api/v1/health` always returns:

```json
{"status": "ok"}
```

It does not verify whether models, MediaPipe assets, or inference dependencies are available.

Relevant file:

- `backend/app/api/endpoints/health.py`

Impact:

- Docker and frontend can treat the backend as healthy while inference is broken.
- Debugging becomes harder because service health does not reflect prediction readiness.

### Backend Dependencies Missing In Current Local Python Environment

In the checked environment, importing the backend fails because packages like `structlog`, `fastapi`, and `torch` are not installed for the active Python interpreter.

Observed failures:

```text
ModuleNotFoundError: No module named 'structlog'
ModuleNotFoundError: No module named 'fastapi'
ModuleNotFoundError: No module named 'torch'
```

Relevant files:

- `backend/requirements.txt`
- `backend/app/main.py`

Impact:

- Running `uvicorn app.main:app` locally fails unless dependencies are installed in the correct environment.
- Python version compatibility may also matter because the current interpreter was Python 3.14, while many ML packages are usually validated on earlier Python versions.

## Prediction Accuracy Issues

### RGB/BGR Face Crop Confusion

Images are loaded through PIL and converted to RGB, then passed to `detect_largest_face()`. That function defaults to treating 3-channel input as BGR and converts it using OpenCV's `BGR -> RGB` conversion.

Relevant files:

- `backend/app/core/pipeline.py`
- `backend/app/preprocessing/face_detection.py`

Impact:

- Cropped face images may have red and blue channels swapped.
- EfficientNet and Xception face-model predictions can become unreliable.
- This is one of the most likely causes of incorrect image classification.

### Missing Models Are Treated As Soft Warnings

If model weights are absent, the registry logs a warning and continues.

Relevant files:

- `backend/app/models/registry.py`
- `backend/app/core/pipeline.py`

Impact:

- The API can return a prediction with fewer than three learned models.
- With zero models loaded, the ensemble returns `UNCERTAIN` with `0.5` fake probability.
- The frontend still displays the result as if a full pipeline ran.

### Incompatible Checkpoints May Still Load

`load_state_dict(..., strict=False)` allows missing or unexpected checkpoint keys. The code prints those keys but does not fail the load.

Relevant file:

- `backend/app/utils/model_loading.py`

Impact:

- Wrong checkpoint files may produce meaningless predictions.
- A model can appear loaded while using partially initialized or mismatched weights.

### Assumed Class Index Ordering

EfficientNet and Xception assume:

```python
probs[0] = real
probs[1] = fake
```

Relevant files:

- `backend/app/models/efficientnet.py`
- `backend/app/models/xception.py`

Impact:

- If the checkpoint was trained with the opposite class order, predictions will be inverted.
- This can directly explain real images being marked fake or fake images being marked real.

### DIRE And FreqNet Are Advertised But Not Active

The README and frontend mention DIRE/FreqNet, but the registry loads only:

- `univfd`
- `efficientnet`
- `xception`

DIRE is commented out in the model registry.

Relevant files:

- `README.md`
- `frontend/src/pages/UploadPage.tsx`
- `frontend/src/pages/ResultsPage.tsx`
- `backend/app/models/registry.py`
- `backend/app/core/fake_type_classifier.py`

Impact:

- Diffusion classification is weaker than advertised.
- The fake-type classifier uses a default DIRE value of `0.5`, which can distort results.
- Users see model capabilities that are not actually running.

### Heuristic Scores Are Not Calibrated

Several forensic features are hardcoded heuristics:

- PRNU correlation
- Frequency anomaly score
- Compression/ELA score
- Fake-type classification
- PCA centroids

Relevant files:

- `backend/app/preprocessing/prnu.py`
- `backend/app/preprocessing/frequency.py`
- `backend/app/preprocessing/compression.py`
- `backend/app/core/fake_type_classifier.py`
- `backend/app/core/pipeline.py`

Impact:

- Scores may look precise but may not be statistically validated.
- The final verdict may be overconfident for images outside the assumed dataset.

## Backend/API Issues

### `/detect` Lacks Top-Level Exception Handling

`pipeline.run()` is called directly without wrapping unexpected model or analyzer errors.

Relevant file:

- `backend/app/api/endpoints/detection.py`

Impact:

- User-facing failures become generic HTTP 500 errors.
- Frontend cannot show actionable messages.

### Frontend Timeout May Be Too Short For CPU Inference

The frontend API client uses a 120-second timeout.

Relevant file:

- `frontend/src/api/client.ts`

Impact:

- CPU inference with CLIP, EfficientNet, Xception, MediaPipe, and analyzers can exceed the timeout on slower machines.
- The frontend may report a network or timeout error even though the backend is still working.

### `models_dir` Setting Is Ignored By Registry Paths

`ModelRegistry.load_all(models_dir, device)` accepts `models_dir`, but actual model paths come from `MODEL_PATHS`.

Relevant files:

- `backend/app/models/registry.py`
- `backend/app/config/settings.py`

Impact:

- Docker/local model configuration is confusing.
- Setting `MODELS_DIR` alone may not point the registry to the expected files.

### CORS Is Too Broad For Deployment

The backend allows all origins, methods, and headers while also enabling credentials.

Relevant file:

- `backend/app/main.py`

Impact:

- Not the cause of local prediction failure, but unsafe for production.

## Frontend And Connection Issues

### Error Messages Are Too Generic

The Zustand store displays `err.message`, which often becomes `Network Error` or timeout text instead of the backend's actual `detail`.

Relevant file:

- `frontend/src/store/detectionStore.ts`

Impact:

- Users do not see whether the problem is missing models, invalid image type, file size, timeout, or backend crash.

### Docker Compose Runtime Env Is Misleading For Vite

`VITE_API_URL` is set in `docker-compose.yml` under the frontend service, but Vite environment variables are baked during build.

Relevant files:

- `docker-compose.yml`
- `frontend/src/api/client.ts`
- `frontend/nginx.conf`

Impact:

- The runtime environment variable does not behave as it appears.
- Docker currently works mostly because the frontend defaults to `/api/v1` and nginx proxies `/api/` to the backend.

### `depends_on` Does Not Wait For Backend Readiness

The frontend depends on the backend container, but this does not guarantee the backend is ready for inference.

Relevant file:

- `docker-compose.yml`

Impact:

- The frontend can be available before backend startup and model loading are finished.
- Users may submit images too early and see connection errors.

## Visualization Issues

### Heatmaps Can Be Misaligned

The backend generates heatmaps using the cropped face image, then resizes the heatmap over the full original image.

Relevant file:

- `backend/app/api/endpoints/visualization.py`

Impact:

- Highlighted regions may appear in the wrong part of the image.
- Visual explanations can be misleading.

### Facial Landmarks Can Be Misaligned

Landmarks are computed on `working_image`, which may be a cropped face, but drawn over the full original preview in the frontend.

Relevant files:

- `backend/app/core/pipeline.py`
- `frontend/src/components/heatmap/HeatmapViewer.tsx`

Impact:

- Green facial landmark dots may not match the original image coordinates.
- This reduces trust in the visualization.

### Heatmap Failures Are Silent In UI

If a heatmap request fails, the frontend only stops the spinner. It does not show an error message.

Relevant file:

- `frontend/src/components/heatmap/HeatmapViewer.tsx`

Impact:

- Users cannot tell whether the heatmap failed because a model is unloaded, unsupported, timed out, or cache data expired.

### Backend Allows Heatmap Names For Models That Are Not Loaded

The visualization endpoint allows:

- `freqnet`
- `dire`

but those models are not loaded by the registry.

Relevant file:

- `backend/app/api/endpoints/visualization.py`

Impact:

- Requests for those heatmaps return service errors.
- API behavior does not match model availability.

## Repository And Documentation Issues

### README Is Stale

The README mentions:

- Missing `models/download_weights.py`
- Old `analysis/` directory
- FreqNet and DIRE as active models
- Project name/path `deepfake-detector`

Relevant file:

- `README.md`

Impact:

- Setup instructions are not reliable.
- New users are likely to run incorrect commands.

### Dataset Images Are Tracked In Git

The repo contains about 120,000 tracked image files under `images/`, using roughly 469 MB locally.

Impact:

- Clone, push, and pull operations are heavy.
- Git history becomes large.
- The source repository mixes code and dataset artifacts.

### Generated Python Cache Files Exist Locally

The workspace contains many `__pycache__` and `.pyc` files, including Python 3.14 cache files.

Relevant files:

- `backend/app/**/__pycache__/`

Impact:

- These should remain untracked.
- They indicate local execution artifacts are mixed into the workspace.

## Verification Performed

The following lightweight checks were performed:

- Python syntax compilation for `backend/app` passed.
- Frontend production build passed.
- Backend import failed in the active Python environment due to missing dependencies.
- No `models/` directory or model weights were found.
- No `models/download_weights.py` script was found.
- Repository contains approximately 120,000 tracked image files.

## Recommended Priority Order

1. Add a real model setup path: provide the missing download script or document exact model files and paths.
2. Make backend readiness honest: `/health` or a new `/ready` endpoint should report model and MediaPipe asset status.
3. Prevent silent degraded predictions: fail clearly if required models/assets are missing.
4. Fix RGB/BGR handling in face detection and prediction preprocessing.
5. Validate checkpoint compatibility and class ordering.
6. Align heatmaps and landmarks by tracking crop coordinates back to the original image.
7. Improve frontend error handling to show backend `detail` messages.
8. Clean README to match the actual code.
9. Move dataset images out of Git or use a dataset/storage strategy.

