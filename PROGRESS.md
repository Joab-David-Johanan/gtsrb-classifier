# PyTorch Image Classification — GTSRB

**Goal:** End-to-End PyTorch project application  
**Dataset:** German Traffic Sign Recognition Benchmark (GTSRB)  


---

## Step 1 — Project Setup
- [x] Check if `uv` is installed: `uv --version`
- [x] If not installed, run: `pip install uv`
- [x] Created `pyproject.toml` — defines package, dependencies, and cu130 PyTorch index
- [x] Run `uv sync` to create `.venv` and install dependencies
- [x] Verify torch is installed: `uv pip show torch`

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

## Version Control
- [x] Created `.gitignore` — excludes `.venv/`, `data/`, `checkpoints/`, notebook outputs
- [x] `uv.lock` is committed — pins exact dependency versions for reproducibility

### Commands
```bash
# 1. Initialise git locally
git init
git add .
git commit -m "initial project setup: pyproject.toml, directory structure, exploration notebook"

# 2a. Create GitHub repo and push (requires gh CLI — cli.github.com)
gh repo create gtsrb-classifier --public --source=. --remote=origin --push

# 2b. Or manually if no gh CLI
git remote add origin https://github.com/YOUR_USERNAME/gtsrb-classifier.git
git branch -M main
git push -u origin main
```

## GitHub Industry Practices

### 1. Branching Strategy
- [x] Renamed default branch from `master` to `main`
- [ ] Use trunk-based development — `main` is always deployable
- [ ] Create short-lived feature branches, merge via PR
- Branch naming convention:
```
feature/custom-dataset
feature/training-loop
fix/class-imbalance-sampler
experiment/efficientnet-backbone
```

#### Commands to rename master → main
```bash
# 1. Rename locally
git branch -m master main

# 2. Push new main branch
git push -u origin main

# 3. On GitHub → Settings → General → Default branch → switch to main → Update

# 4. Delete old master on remote (only works AFTER step 3)
git push origin --delete master
```

### 2. Conventional Commits
Every commit message follows this format:
```
feat: add custom GTSRB dataset class
fix: correct WeightedRandomSampler weights
refactor: split trainer into separate module
docs: update config.yaml comments
```

### 3. Branch Protection on `main`
- [x] GitHub → Settings → Branches → Add rule:
  - Require PR before merging
  - Require at least 1 review
  - Require status checks (CI) to pass

### 4. GitHub Actions CI
- [x] Created `.github/workflows/ci.yml`
  - Runs `ruff` linting on every push
  - Runs `pytest` on every push
  - Fails the PR if either fails
  - Uses CPU-only PyTorch in CI (fast + free — no GPU runner needed)

#### Steps to enable status checks in branch protection
```
1. Commit and push .github/workflows/ci.yml to main first
   git add .github/ PROGRESS.md
   git commit -m "ci: add GitHub Actions workflow and PR template"
   git push

2. Wait for CI to run at least once (check the Actions tab on GitHub)

3. Go to Settings → Branches → Add rule → Branch name: main
   - Check: Require a pull request before merging
   - Check: Require status checks to pass → search for "Lint" and "Test"
     (they only appear after CI has run at least once)
   - Save
```

### 5. PR Description Template
- [x] Created `.github/pull_request_template.md`
```
## What does this PR do?

## How to test it?

## Checklist
- [ ] Tests added
- [ ] Config updated if needed
```

### 6. GitHub Issues + Labels
- [x] Open an issue for every feature before starting work
- [x] Label it (e.g. `enhancement`, `bug`, `documentation`)
- [x] Reference the issue in commits to auto-close it when PR merges

#### Full workflow for every new feature
```bash
# 1. Open issue on GitHub (website or CLI)
gh issue create --title "feat: data pipeline" --body "Build custom Dataset, DataLoader, and augmentation pipeline"
# GitHub assigns it a number e.g. #1

# 2. Create a branch named after the feature
git checkout -b feature/data-pipeline

# 3. Commit messages reference the issue number
git commit -m "feat: add custom GTSRB dataset class (closes #1)"
# Using "closes #N" auto-closes the issue when the PR merges to main

# 4. Push the branch and open a PR
git push -u origin feature/data-pipeline
gh pr create --title "feat: data pipeline" --body "Closes #1"
```

### What to set up now vs later

| Now | Later |
|---|---|
| Branch protection on `main` | GitHub Projects board |
| Conventional commits habit | Semantic versioning + tags |
| CI with ruff + pytest | Automated model benchmarking in CI |
| PR template | Dependabot for dep updates |

## Logging & Error Handling
- [x] Added `loguru` to `pyproject.toml` core dependencies
- [x] Created `src/gtsrb/utils/logger.py` — single loguru config imported everywhere
- [x] Created `src/gtsrb/utils/exceptions.py` — custom exception hierarchy
- [x] Created `src/gtsrb/utils/__init__.py` — exports logger and all exceptions

### How to use in any file
```python
from gtsrb.utils import logger, DatasetError

logger.info("Loading dataset split: train")
logger.debug("Image shape: {shape}", shape=img.shape)

raise DatasetError("Class folder 00005 is empty")
```

### Exception hierarchy
```
GtsrbError          ← base, catch-all
├── DatasetError    ← data loading / processing
├── ConfigError     ← missing or invalid config values
├── CheckpointError ← saving / loading model weights
├── ModelError      ← architecture or forward pass
└── AugmentationError ← augmentation pipeline
```

### Log output format
```
2026-04-29 10:42:01 | INFO     | gtsrb.data.dataset:__getitem__:47 | Loading image for class 12
2026-04-29 10:42:01 | ERROR    | gtsrb.training.trainer:train:83   | DataLoader returned empty batch
```
Logs also written to `logs/gtsrb.log` with 10 MB rotation and 7-day retention.

## Step 2 — Data Pipeline
- [x] Created `notebooks/01_explore_data.ipynb` — download dataset, inspect structure, plot class distribution, visualise samples, check image sizes
- [x] Created `src/gtsrb/data/transforms.py` — train augmentations + val transforms
- [x] Created `src/gtsrb/data/dataset.py` — custom Dataset class
- [x] Created `src/gtsrb/data/dataloader.py` — DataLoader factory with WeightedRandomSampler

### To run the notebook
```bash
uv sync --extra dev     # installs matplotlib, jupyter, ipykernel
jupyter notebook notebooks/01_explore_data.ipynb
```

### Key concepts in the data pipeline

**transforms.py**
- Train: heavy augmentations (blur, brightness, perspective, rotation) — teaches model to handle real-world variation
- Val: only resize + normalize — keeps evaluation metrics consistent and comparable across epochs

**dataset.py**
- Stores only `(path, label)` pairs at init — images loaded lazily in `__getitem__` to avoid RAM overflow
- `image.convert("RGB")` guards against rare grayscale images breaking the pipeline
- `get_class_counts()` returns per-class image counts — used by DataLoader to fix imbalance

**dataloader.py**
- `WeightedRandomSampler` replaces `shuffle=True` for training — gives rare classes equal chance of appearing in each batch
- `pin_memory=True` pre-loads batches into pinned RAM for faster CPU→GPU transfer
