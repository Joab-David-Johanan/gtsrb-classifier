# PyTorch Image Classification — GTSRB

**Goal:** End-to-End PyTorch project application  
**Dataset:** German Traffic Sign Recognition Benchmark (GTSRB)  


---

## Step 1 — Project Setup
- [x] Check if `uv` is installed: `uv --version`
- [x] If not installed, run: `pip install uv`
- [x] Run `uv sync` to create `.venv` and install dependencies

### Files Created & Why

| File / Folder | Reason |
|---|---|
| `pyproject.toml` | Single source of truth for the package — defines dependencies, build system, and optional groups (dev/serving/tracking). Replaces `requirements.txt`. Makes the project installable on any machine with `uv sync`. |
| `configs/config.yaml` | Centralises all hyperparameters and paths. Nothing is hardcoded in Python files — changing a value here affects the whole project without touching code. |
| `src/gtsrb/` | The installable package lives under `src/` so Python cannot accidentally import it without it being installed first — prevents subtle path bugs in production. |
| `src/gtsrb/data/` | Will hold the custom `Dataset` class and augmentation pipeline — kept separate so data logic is never mixed with model or training logic. |
| `src/gtsrb/models/` | Will hold the backbone + classification head definition — isolated so you can swap ResNet for EfficientNet without touching training code. |
| `src/gtsrb/training/` | Will hold the training loop, loss functions, and scheduler logic — separated so the same trainer can be reused with different models or datasets. |
| `src/gtsrb/serving/` | Will hold the FastAPI inference endpoint — completely decoupled from training so it can be deployed independently as a Docker container. |
| `scripts/` | Entry-point scripts (`train.py`, `evaluate.py`) that wire config → model → trainer together. Kept thin — all real logic lives in `src/gtsrb/`. |
| `tests/` | Unit and integration tests. Separating tests from source is standard packaging practice and required by most CI systems. |

## Step 2 — Data Pipeline
- [x] Created `notebooks/01_explore_data.ipynb` — download dataset, inspect structure, plot class distribution, visualise samples, check image sizes

### To run the notebook
```bash
uv sync --extra dev     # installs matplotlib, jupyter, ipykernel
jupyter notebook notebooks/01_explore_data.ipynb
```
