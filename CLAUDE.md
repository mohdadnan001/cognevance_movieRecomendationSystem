# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Collaborative-filtering movie recommender on MovieLens ml-latest-small, with a Streamlit UI. Models are scikit-surprise SVD and item-based k-NN `KNNBasic`; a popularity baseline serves cold-start users.

The git repo root is this directory (the parent `movieRecomendationSystem/` folder is just a container). Python 3.12 venv lives in `.venv/`.

## Commands

All commands run from the repo root.

```bash
# Tests (integration tests — see caveat below)
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pytest tests/test_preprocessing.py::test_clean_movies -v

# Regenerate preprocessing artifacts from data/raw (writes data/processed/)
.venv/Scripts/python.exe -m src.preprocessing

# Retrain + save SVD/k-NN (grid search, slow — many minutes)
.venv/Scripts/python.exe -m src.recommender

# Ad-hoc checks (each has a __main__ block that loads models)
.venv/Scripts/python.exe scripts/evaluate_models.py
.venv/Scripts/python.exe scripts/test_recommendation_engine.py
.venv/Scripts/python.exe scripts/test_streamlit_imports.py

# App
.venv/Scripts/python.exe -m streamlit run app/app.py   # or: streamlit run app/app.py
```

**Tests are integration tests, not unit tests.** They read `data/processed/` artifacts (`mappings.joblib`, `trainset.joblib`, `models/algo_svd.pkl`, `models/algo_knn.pkl`), which are gitignored. On a fresh clone, run the preprocessing and training commands first or every test errors on missing files.

## Architecture

Pipeline stages, each with a matching notebook in `notebooks/` (`01_eda`, `02_preprocessing`, `03_model_training`):

1. `src/data_loader.py` — reads the four MovieLens CSVs straight from `data/raw/*.csv` (not the nested `data/raw/ml-latest-small/` copy, which is a redundant unzip). Parquet helpers for `data/processed/`.
2. `src/preprocessing.py` — clean movies (parses `genres_list`, `year`, `clean_title`), filters to ≥5 ratings per user/movie, builds the sparse user-item matrix and `mappings`, user-based or time-based train/test split, saves everything to `data/processed/`.
3. `src/recommender.py` — surprise `GridSearchCV` training for SVD/k-NN, persists best estimators to `data/processed/models/`, plus inner-ID → ranked `(item_inner_id, est)` helpers.
4. `src/evaluation.py` — RMSE/MAE plus Precision@K, Recall@K, Coverage@K, Novelty@K, and a popularity baseline.
5. `src/recommendation_engine.py` — the app-facing layer: turns ranked inner IDs into metadata dicts (`title`, `genres`, `year`, `imdbId`, `tmdbId`, `estimated_rating`), poster fetching from OMDb with placeholder fallback, cold-start and rating-history helpers.
6. `app/app.py` + `app/utils.py` — Streamlit UI. Three modes (User ID / Simulated User / Popular); `app/utils.py` holds poster fetching with `@st.cache_data` and display/layout helpers.

`src/__init__.py` re-exports the public API, but internal modules import as `from src.xxx import ...`, so code must execute with the repo root on `sys.path`.

### Import conventions differ by location (easy to trip on)

- `src/` modules: absolute imports (`from src.preprocessing import ...`).
- `tests/`: `sys.path.insert(0, .../src)` then `from preprocessing import ...` — **without** the `src.` prefix.
- `app/app.py` inserts the repo root into `sys.path` before importing `src.*`.
- `notebooks/` and `scripts/`: `sys.path.append('..')` / `sys.path.append('.')`, so they only work when the notebook/script is launched from its own directory.

### OMDb / poster handling

API key resolution is duplicated in two places and must stay in sync: `src/recommendation_engine.py` reads `.env` only, `app/utils.py` prefers `st.secrets["OMDB_API_KEY"]` then falls back to `.env`. Both have their own poster fetch function (`fetch_poster_omdb`, `fetch_poster`) and their own `_placeholder_poster`. The app's actual grid rendering uses `app/utils.fetch_poster`; the engine's copy only runs when calling `get_user_recommendations_with_metadata(..., include_posters=True)`.

### ID mapping pitfall (known, unfixed)

Two independent inner-ID spaces exist: `mappings.joblib` (`idx_to_movie`) is built from all filtered ratings, while the surprise `trainset` builds its own inner IDs from `train_data`. `recommendation_engine` converts recommendation inner IDs via `idx_to_movie`, which does not line up with the trainset's `to_raw_iid` (verified: 2,268 of 2,269 indices disagree). If you touch ranking/metadata mapping, use `trainset.to_raw_iid(iid_inner)` instead of `idx_to_movie[iid_inner]`.

### Other current-state notes

- `app/app.py` in the Simulated User mode is a stub: it selects movies but recommends using fixed user 1's SVD recommendations (`TODOs` in that block).
- The current uncommitted copy of `app/app.py` line ~205 does `with cols:` where `cols` is a list of Streamlit columns — that raises `TypeError: 'list' object does not support the context manager protocol` (repeats in `streamlit.err`). Use `with col:`.
- Metrics shown in the sidebar are hardcoded literals in `app/app.py`, not computed at runtime.

## Configuration

- `.streamlit/config.toml` — dark theme, port 8501. Its `enableCORS=false` is ignored because `enableXsrfProtection=true`; that warning is expected.
- `.streamlit/secrets.toml` and `.env` hold `OMDB_API_KEY` (both gitignored).
- `packages.txt` exists for Streamlit Community Cloud system deps.
