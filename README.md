# Movie Recommender System

Content-based movie recommendation system built with Python, pandas, scikit-learn, and Streamlit. Uses MovieLens ml-latest-small dataset.

## Project Overview

| Aspect | Details |
|--------|---------|
| **Approach** | Content-based filtering (TF-IDF + Cosine Similarity) |
| **Dataset** | MovieLens ml-latest-small (9,742 movies, 100,836 ratings) |
| **Stack** | Python, pandas, scikit-learn, Streamlit, OMDb API |
| **Deployment** | Streamlit Community Cloud |

## Project Structure

```
movie-recommender/
├── data/
│   ├── raw/              # Original MovieLens CSV files
│   └── processed/        # Cleaned data, pickled matrices (gitignored)
├── notebooks/
│   └── 01_eda.ipynb      # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # Load raw/processed data
│   ├── preprocessing.py  # Data cleaning & feature engineering (Day 2)
│   ├── recommender.py    # TF-IDF, similarity, recommendations (Day 3-4)
│   └── evaluation.py     # Precision@K, coverage metrics (Day 5)
├── app/
│   ├── __init__.py
│   ├── app.py            # Streamlit UI (Day 6-7)
│   └── utils.py          # Poster fetching, helpers
├── tests/
│   └── test_recommender.py
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
- [x] Create virtual environment & install all dependencies
- [x] Verify data loading works (9,742 movies, 100,836 ratings)

### Day 2 - Data Cleaning & EDA 🔄
- [ ] Run EDA notebook to generate insights
- [ ] Build `src/preprocessing.py` with cleaning logic
- [ ] Merge movies + ratings, filter movies with ≥5 ratings
- [ ] Handle missing values, parse genres
- [ ] Save cleaned data to `data/processed/`

### Day 3 - Feature Engineering
- [ ] TF-IDF vectorization on genres + overview
- [ ] Save vectorizer and TF-IDF matrix

### Day 4 - Model Building
- [ ] Compute cosine similarity matrix
- [ ] Implement `get_recommendations(movie_title, top_n=10)`

### Day 5 - Evaluation
- [ ] Train/test split by user
- [ ] Precision@10 vs. popularity baseline
- [ ] Coverage & diversity metrics

### Day 6-7 - Streamlit UI
- [ ] Search interface, recommendations grid
- [ ] OMDb API integration for movie posters
- [ ] Caching, error handling, styling

### Day 8 - Documentation
- [ ] Complete README with architecture diagram
- [ ] Record demo GIF

### Day 9 - Deployment
- [ ] Deploy to Streamlit Community Cloud
- [ ] Verify live app

### Day 10 - Stretch Goal (Optional)
- [ ] Collaborative filtering branch (SVD/user-item)

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

## Future Improvements

- Hybrid recommender (content + collaborative)
- User profiles & personalization
- Real-time model retraining
- A/B testing framework
- Docker containerization

## License

MIT License - see LICENSE file for details.