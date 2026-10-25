"""Data preprocessing for collaborative filtering."""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import sparse
import joblib
from surprise import Dataset, Reader

from src.data_loader import load_raw_data, save_processed, load_processed

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Clean movies dataframe."""
    df = movies.copy()
    # Parse genres
    df["genres_list"] = df["genres"].str.split("|")
    # Handle "(no genres listed)"
    df["genres_list"] = df["genres_list"].apply(
        lambda x: [] if x == ["(no genres listed)"] else x
    )
    # Extract year from title
    df["year"] = df["title"].str.extract(r"\((\d{4})\)", expand=False)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    # Clean title (remove year)
    df["clean_title"] = df["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    return df


def filter_ratings(ratings: pd.DataFrame, min_movie_ratings: int = 5, min_user_ratings: int = 5) -> pd.DataFrame:
    """Filter ratings to keep only movies/users with sufficient ratings."""
    df = ratings.copy()
    
    # Filter movies with at least min_movie_ratings
    movie_counts = df.groupby("movieId").size()
    valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
    df = df[df["movieId"].isin(valid_movies)]
    
    # Filter users with at least min_user_ratings
    user_counts = df.groupby("userId").size()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    df = df[df["userId"].isin(valid_users)]
    
    print(f"After filtering: {len(df):,} ratings, {df['userId'].nunique()} users, {df['movieId'].nunique()} movies")
    return df


def create_user_item_matrix(ratings: pd.DataFrame) -> sparse.csr_matrix:
    """Create sparse user-item matrix from ratings."""
    # Map userId and movieId to contiguous indices
    user_ids = ratings["userId"].unique()
    movie_ids = ratings["movieId"].unique()
    
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}
    idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
    idx_to_movie = {idx: mid for mid, idx in movie_to_idx.items()}
    
    # Create sparse matrix
    rows = ratings["userId"].map(user_to_idx).values
    cols = ratings["movieId"].map(movie_to_idx).values
    data = ratings["rating"].values
    
    matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(movie_ids))
    )
    
    mappings = {
        "user_to_idx": user_to_idx,
        "movie_to_idx": movie_to_idx,
        "idx_to_user": idx_to_user,
        "idx_to_movie": idx_to_movie,
        "user_ids": user_ids,
        "movie_ids": movie_ids,
    }
    
    return matrix, mappings


def create_surprise_dataset(ratings: pd.DataFrame) -> Dataset:
    """Create surprise Dataset from ratings."""
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    return data


def user_based_train_test_split(ratings: pd.DataFrame, test_ratio: float = 0.2, random_state: int = 42):
    """
    Split ratings by user: hold out last N ratings per user for test.
    This simulates predicting future ratings based on past.
    """
    np.random.seed(random_state)
    
    # Sort by timestamp per user
    ratings_sorted = ratings.sort_values(["userId", "timestamp"])
    
    train_rows = []
    test_rows = []
    
    for user_id, group in ratings_sorted.groupby("userId"):
        n_ratings = len(group)
        n_test = max(1, int(n_ratings * test_ratio))
        
        # Hold out last n_test ratings for test
        test_rows.append(group.iloc[-n_test:])
        train_rows.append(group.iloc[:-n_test])
    
    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)
    
    print(f"Train: {len(train_df):,} ratings, Test: {len(test_df):,} ratings")
    print(f"Train users: {train_df['userId'].nunique()}, Test users: {test_df['userId'].nunique()}")
    
    return train_df, test_df


def time_based_train_test_split(ratings: pd.DataFrame, test_ratio: float = 0.2):
    """Split ratings by time: earlier ratings for train, later for test."""
    ratings_sorted = ratings.sort_values("timestamp")
    split_idx = int(len(ratings_sorted) * (1 - test_ratio))
    
    train_df = ratings_sorted.iloc[:split_idx].copy()
    test_df = ratings_sorted.iloc[split_idx:].copy()
    
    print(f"Train: {len(train_df):,} ratings (up to {train_df['timestamp'].max()})")
    print(f"Test: {len(test_df):,} ratings (from {test_df['timestamp'].min()})")
    
    return train_df, test_df


def save_preprocessed_data(
    movies_clean: pd.DataFrame,
    ratings_clean: pd.DataFrame,
    user_item_matrix: sparse.csr_matrix,
    mappings: dict,
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
):
    """Save all preprocessed data."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save DataFrames as parquet
    movies_clean.to_parquet(PROCESSED_DIR / "movies_clean.parquet", index=False)
    ratings_clean.to_parquet(PROCESSED_DIR / "ratings_clean.parquet", index=False)
    train_ratings.to_parquet(PROCESSED_DIR / "train_ratings.parquet", index=False)
    test_ratings.to_parquet(PROCESSED_DIR / "test_ratings.parquet", index=False)
    
    # Save sparse matrix
    sparse.save_npz(PROCESSED_DIR / "user_item_matrix.npz", user_item_matrix)
    
    # Save mappings
    joblib.dump(mappings, PROCESSED_DIR / "mappings.joblib")
    
    # Save surprise datasets
    train_data = create_surprise_dataset(train_ratings)
    test_data = create_surprise_dataset(test_ratings)
    joblib.dump(train_data, PROCESSED_DIR / "trainset.joblib")
    joblib.dump(test_data, PROCESSED_DIR / "testset.joblib")
    
    print(f"All preprocessed data saved to {PROCESSED_DIR}")


def load_preprocessed_data():
    """Load all preprocessed data."""
    movies_clean = load_processed("movies_clean")
    ratings_clean = load_processed("ratings_clean")
    train_ratings = load_processed("train_ratings")
    test_ratings = load_processed("test_ratings")
    user_item_matrix = sparse.load_npz(PROCESSED_DIR / "user_item_matrix.npz")
    mappings = joblib.load(PROCESSED_DIR / "mappings.joblib")
    train_data = joblib.load(PROCESSED_DIR / "trainset.joblib")
    test_data = joblib.load(PROCESSED_DIR / "testset.joblib")
    
    return {
        "movies_clean": movies_clean,
        "ratings_clean": ratings_clean,
        "train_ratings": train_ratings,
        "test_ratings": test_ratings,
        "user_item_matrix": user_item_matrix,
        "mappings": mappings,
        "train_data": train_data,
        "test_data": test_data,
    }


def run_preprocessing_pipeline(
    min_movie_ratings: int = 5,
    min_user_ratings: int = 5,
    test_ratio: float = 0.2,
    split_method: str = "user_based",
    random_state: int = 42,
):
    """Run the full preprocessing pipeline."""
    print("Loading raw data...")
    raw_data = load_raw_data()
    movies = raw_data["movies"]
    ratings = raw_data["ratings"]
    
    print("Cleaning movies...")
    movies_clean = clean_movies(movies)
    
    print("Filtering ratings...")
    ratings_clean = filter_ratings(ratings, min_movie_ratings, min_user_ratings)
    
    # Merge to get movie titles in ratings
    ratings_clean = ratings_clean.merge(
        movies_clean[["movieId", "clean_title", "genres_list", "year"]],
        on="movieId",
        how="left"
    )
    
    print("Creating user-item matrix...")
    user_item_matrix, mappings = create_user_item_matrix(ratings_clean)
    
    print(f"Creating train/test split ({split_method})...")
    if split_method == "user_based":
        train_ratings, test_ratings = user_based_train_test_split(
            ratings_clean, test_ratio, random_state
        )
    elif split_method == "time_based":
        train_ratings, test_ratings = time_based_train_test_split(ratings_clean, test_ratio)
    else:
        raise ValueError(f"Unknown split_method: {split_method}")
    
    print("Saving preprocessed data...")
    save_preprocessed_data(
        movies_clean, ratings_clean, user_item_matrix, mappings,
        train_ratings, test_ratings
    )
    
    return {
        "movies_clean": movies_clean,
        "ratings_clean": ratings_clean,
        "user_item_matrix": user_item_matrix,
        "mappings": mappings,
        "train_ratings": train_ratings,
        "test_ratings": test_ratings,
    }


if __name__ == "__main__":
    run_preprocessing_pipeline()