# City Hacks Streetscape Simulation

AI-assisted sidewalk accessibility analysis platform with two integrated workflows:

1. **Map Audit**: scans GeoJSON sidewalk assets, flags ADA-related violations, and visualizes results on an interactive map.
2. **Image AI Advisor**: predicts sidewalk condition (`Good`, `Fair`, `Poor`) from an uploaded photo and generates actionable recommendations with Gemini.

## Innovation Summary

This project combines:

- **Rule-based civic geospatial auditing** (Brookline sidewalk asset data + obstacle checks)
- **Computer vision quality scoring** (CNN sidewalk-condition classifier)
- **LLM-grounded recommendations** (Gemini produces structured improvement actions)
- **Safety gate for bad inputs** (Gemini first verifies a real sidewalk is present; if not, classification is skipped)

This makes the output practical for municipal operations: find risky segments on a map, then use image-level AI advice for maintenance actions.

## Repository

- GitHub: [City-hacks-streetscape-simulation](https://github.com/GeesalaPrashanthi/City-hacks-streetscape-simulation)
- Main app files:
  - `main.py` (FastAPI backend)
  - `frontend/src/App.jsx` (React UI, 2 tabs)
  - `train.py` (baseline training)
  - `train_advanced.py` (fair/advanced training)
  - `download_images.py` (dataset ingestion)
  - `mask_images.py` (sidewalk masking)

## What Is Included vs Excluded

Included in Git:
- source code
- GeoJSON/metadata files
- training scripts and result JSONs
- empty dataset folder structure via `.gitkeep`

Excluded from Git (by `.gitignore`):
- raw dataset images in `dataset/**`
- masked dataset images in `dataset_masked/**`
- model checkpoints (`*.pt`, `*.pth`, etc.)
- local virtual environments and secret env files

## System Requirements

- Python 3.10+
- Node.js 18+
- npm 9+
- macOS/Linux/Windows (MPS/CUDA/CPU supported by torch)

## Quick Start

### 1. Clone

```bash
git clone https://github.com/GeesalaPrashanthi/City-hacks-streetscape-simulation.git
cd City-hacks-streetscape-simulation
```

### 2. Backend setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn geopandas torch torchvision pillow numpy transformers
```

If your environment needs full GeoPandas stack support, install with conda/mamba (recommended on some systems).

### 3. Configure environment variables

```bash
export GEMINI_API_KEY="your_google_ai_studio_key"
export GEMINI_MODEL="gemini-3.1-pro-preview"
# optional fallback models (comma-separated)
export GEMINI_FALLBACK_MODELS="gemini-flash-latest,gemini-2.5-flash-lite"

# optional: choose which checkpoint to load
export MODEL_PATH="sidewalk_classifier_fair.pt"
```

### 4. Run backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Backend API will be available at `http://localhost:8001`.

### 5. Run frontend

In a second terminal:

```bash
cd frontend
npm install
# optional if backend is not on localhost:8001
export VITE_API_BASE_URL="http://localhost:8001"
npm run dev
```

Open the local Vite URL shown in terminal.

## App Usage

### Tab 1: Map Audit

- Fetches `/sidewalks` from backend.
- Displays compliance summary and filtered markers.
- Shows violations like:
  - missing sidewalk
  - poor condition
  - inaccessible material
  - nearby obstacle proximity

### Tab 2: Image AI Advisor

Upload an image and click **Analyze Sidewalk**.

Pipeline:
1. Gemini checks if the image actually contains a sidewalk.
2. If no sidewalk: classifier is skipped, user gets a clear message.
3. If sidewalk present: classifier predicts `Good/Fair/Poor` + probabilities.
4. Gemini returns structured explanation and 3 improvement actions targeted to the predicted class.

## API Endpoints

- `GET /sidewalks` -> full feature list + summary
- `GET /summary` -> top-level metrics
- `GET /violations` -> non-compliant subset
- `POST /predict-sidewalk` -> image condition prediction + optional Gemini summary

`POST /predict-sidewalk` form fields:
- `image` (file, required)
- `include_gemini` (`true/false`)
- `guidance_prompt` (optional text)
- `enforce_sidewalk_check` (`true/false`)

## Data Preparation Workflow

### 1. Download images from GeoJSON

```bash
python download_images.py --mode full --out-dir dataset
```

Useful options:
- `--mode balanced` for equal per-class download
- `--max-per-class N` to cap class count
- `--skip-existing` (default true)

### 2. Create masked dataset (optional but recommended)

```bash
python mask_images.py
```

Outputs to `dataset_masked/Good`, `dataset_masked/Fair`, `dataset_masked/Poor`.

## Training

### Baseline training

```bash
python train.py
```

### Advanced fair training (recommended)

```bash
python train_advanced.py \
  --data-dir dataset_masked \
  --split-mode fair \
  --equal-train-per-class \
  --weighted-sampler \
  --arch convnext_tiny \
  --epochs 35 \
  --batch-size 16 \
  --num-workers 0 \
  --model-out sidewalk_classifier_fair.pt \
  --results-out training_results_fair.json
```

## Local Prediction (CLI)

```bash
python predict.py /path/to/image.jpg
```

Or URL:

```bash
python predict.py https://example.com/sidewalk.jpg
```

## Troubleshooting

- **Gemini 429 quota error**: free tier limit reached; wait/reset quota or enable billing.
- **Gemini 404 model unavailable**: update `GEMINI_MODEL` to a currently available model for your key/project.
- **Classifier unavailable at startup**: set `MODEL_PATH` to a valid `.pt` checkpoint.
- **Slow dataloading or worker issues on macOS**: use `--num-workers 0`.

## Notes for Contributors

- Do not commit `dataset/**`, `dataset_masked/**`, or checkpoint files.
- Keep only `.gitkeep` placeholders in dataset class folders.
- Use environment variables for all secrets.
