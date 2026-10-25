# Movie Recommender System

Collaborative filtering movie recommendation system built with Python, pandas, scikit-learn, scikit-surprise, and Streamlit. Uses MovieLens ml-latest-small dataset.

## Project Overview

| Aspect | Details |
|--------|---------|
| **Approach** | Collaborative Filtering (SVD / k-NN on user-item matrix) |
| **Dataset** | MovieLens ml-latest-small (9,742 movies, 100,836 ratings, 610 users) |
| **Stack** | Python, pandas, scikit-learn, scikit-surprise, Streamlit, OMDb API |
| **Deployment** | Streamlit Community Cloud |

## Project Structure

```
movie-recommender/
├── data/
│   ├── raw/              # Original MovieLens CSV files
│   └── processed/        # Cleaned data, pickled models (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb      # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # User-item matrix, train/test split
│   ├── 03_model_training.ipynb # SVD, k-NN training & tuning
│   └── 04_evaluation.ipynb     # RMSE, Precision@K, Coverage
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # Load raw/processed data
│   ├── preprocessing.py  # Data cleaning, user-item matrix
│   ├── recommender.py    # CF model training, prediction, recommendations
│   └── evaluation.py     # RMSE, Precision@K, Coverage metrics
├── app/
│   ├── __init__.py
│   ├── app.py            # Streamlit UI (User ID input)
│   └── utils.py          # Poster fetching, helpers
├── tests/
│   ├── test_preprocessing.py
│   ├── test_recommender.py
│   └── test_evaluation.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Progress Tracker

### Day 1 - Setup & Data Loading ✅
- [x] Initialize git repository
- [x] Create project folder structure
- [x] Download MovieLens ml-latest-small dataset
- [x] Create `requirements.txt` with core + dev dependencies
- [x] Create `.gitignore` (ignores processed data, cache, secrets)
- [x] Build `src/data_loader.py` for loading raw/processed data
- [x] Create EDA notebook boilerplate (`notebooks/01_eda.ipynb`)
- [x] Create virtual environment & install all dependencies (including scikit-surprise)
- [x] Verify data loading works (9,742 movies, 100,836 ratings, 610 users)

### Day 2 - Data Cleaning & EDA 🔄
- [ ] Run EDA notebook to generate insights
- [ ] Build `src/preprocessing.py` with cleaning logic
- [ ] Merge movies + ratings, filter movies with ≥5 ratings
- [ ] Handle missing values, parse genres
- [ ] Build user-item sparse matrix
- [ ] Train/test split by user (for CF evaluation)
- [ ] Save cleaned data to `data/processed/`

### Day 3 - User-Item Matrix & Train/Test Split
- [ ] Convert to surprise `Dataset` format
- [ ] Create trainset/testset split (time-based or user-based)
- [ ] Save splits for reproducible training

### Day 4 - Model Training (CF)
- [ ] Train SVD (matrix factorization) with grid search
- [ ] Train k-NN (item-based) as baseline
- [ ] Save best models (`algo_svd.pkl`, `algo_knn.pkl`)

### Day 5 - Evaluation
- [ ] RMSE, MAE on test set
- [ ] Precision@10, Recall@10 vs. popularity baseline
- [ ] Coverage & novelty metrics
- [ ] Compare SVD vs. k-NN

### Day 6 - Recommendation Engine
- [ ] `get_user_recommendations(user_id, top_n=10)`
- [ ] Cold-start fallback: popularity-based for new users
- [ ] Unit tests for recommender

### Day 7 - Streamlit UI
- [ ] User ID input (1-610) → recommendations grid
- [ ] OMDb API integration for movie posters
- [ ] Caching, error handling, loading states

### Day 8 - Polish UI
- [ ] Styling, responsive layout
- [ ] "Simulated user" mode (select rated movies)
- [ ] Model comparison toggle (SVD vs k-NN)

### Day 9 - Documentation & Deploy
- [ ] Update README with CF architecture diagram
- [ ] Record demo GIF
- [ ] Deploy to Streamlit Community Cloud

### Day 10 - Stretch Goal: Hybrid
- [ ] Create `cf-hybrid` branch
- [ ] Blend CF + content-based (weighted)
- [ ] Compare hybrid vs. pure CF

## Quick Start

```bash
# Clone & setup
git clone <repo-url>
cd movie-recommender
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt

# Run EDA
jupyter notebook notebooks/01_eda.ipynb

# Run Streamlit app (after Day 7)
streamlit run app/app.py
```

## Dataset

- **Source**: [GroupLens MovieLens](https://grouplens.org/datasets/movielens/latest/)
- **Version**: ml-latest-small
- **Files**: movies.csv, ratings.csv, links.csv, tags.csv

## Evaluation Metrics

| Metric | Target |
|--------|--------|
| RMSE | < 0.90 |
| MAE | < 0.70 |
| Precision@10 | > 0.15 |
| Coverage | > 0.30 |

## Future Improvements

- Hybrid recommender (CF + content-based)
- Implicit feedback (ALS via `implicit` library)
- Deep learning: Neural CF, AutoEncoders
- Real-time model retraining pipeline
- A/B testing framework
- Docker containerization

## License

MIT License - see LICENSE file for details.