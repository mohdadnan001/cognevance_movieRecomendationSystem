"""Tests for preprocessing module."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import (
    clean_movies,
    filter_ratings,
    create_user_item_matrix,
    create_surprise_dataset,
    user_based_train_test_split,
    time_based_train_test_split,
    run_preprocessing_pipeline,
)


def test_clean_movies():
    """Test movie cleaning."""
    df = pd.DataFrame({
        "movieId": [1, 2, 3],
        "title": ["Toy Story (1995)", "Jumanji (1995)", "No Year Movie"],
        "genres": ["Adventure|Animation", "Adventure|Children", "(no genres listed)"],
    })
    cleaned = clean_movies(df)
    
    assert "genres_list" in cleaned.columns
    assert "year" in cleaned.columns
    assert "clean_title" in cleaned.columns
    assert cleaned.loc[0, "year"] == 1995
    assert cleaned.loc[0, "clean_title"] == "Toy Story"
    assert cleaned.loc[2, "genres_list"] == []
    assert cleaned.loc[0, "genres_list"] == ["Adventure", "Animation"]


def test_filter_ratings():
    """Test rating filtering."""
    # Create test data: 3 users, 4 movies
    # User 1: 3 ratings (movies 1,2,3), User 2: 2 ratings (movies 1,2), User 3: 5 ratings (movies 2,3,4,4,4)
    # Movie 1: 2 ratings (user 1,2), Movie 2: 3 ratings (user 1,2,3), Movie 3: 2 ratings (user 1,3), Movie 4: 3 ratings (user 3)
    df = pd.DataFrame({
        "userId": [1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
        "movieId": [1, 2, 3, 1, 2, 2, 3, 4, 4, 4],
        "rating": [4.0, 5.0, 3.0, 4.0, 3.0, 5.0, 4.0, 5.0, 4.0, 3.0],
        "timestamp": range(10),
    })
    
    # Filter min_movie=2, min_user=3
    # All movies have >=2 ratings, so no movies removed
    # Users 1 and 3 have >=3 ratings, User 2 has only 2 -> removed
    filtered = filter_ratings(df, min_movie_ratings=2, min_user_ratings=3)
    
    # User 2 removed (only 2 ratings), Users 1 and 3 kept
    assert filtered["userId"].nunique() == 2
    assert set(filtered["userId"].unique()) == {1, 3}
    assert len(filtered) == 8  # User 1 (3) + User 3 (5) = 8 ratings
    assert filtered["movieId"].nunique() == 4  # All movies kept (all have >=2 ratings)


def test_create_user_item_matrix():
    """Test user-item matrix creation."""
    df = pd.DataFrame({
        "userId": [1, 1, 2, 2],
        "movieId": [10, 20, 10, 30],
        "rating": [4.0, 5.0, 3.0, 4.0],
    })
    
    matrix, mappings = create_user_item_matrix(df)
    
    assert isinstance(matrix, sparse.csr_matrix)
    assert matrix.shape == (2, 3)  # 2 users, 3 movies
    assert matrix[0, 0] == 4.0  # user 1 -> movie 10
    assert matrix[0, 1] == 5.0  # user 1 -> movie 20
    assert matrix[1, 0] == 3.0  # user 2 -> movie 10
    assert matrix[1, 2] == 4.0  # user 2 -> movie 30
    assert "user_to_idx" in mappings
    assert "movie_to_idx" in mappings
    assert "idx_to_user" in mappings
    assert "idx_to_movie" in mappings


def test_create_surprise_dataset():
    """Test surprise dataset creation."""
    df = pd.DataFrame({
        "userId": [1, 1, 2],
        "movieId": [10, 20, 10],
        "rating": [4.0, 5.0, 3.0],
    })
    
    data = create_surprise_dataset(df)
    
    from surprise import Dataset
    assert isinstance(data, Dataset)
    # Check we can build trainset
    trainset = data.build_full_trainset()
    assert trainset.n_users == 2
    assert trainset.n_items == 2
    assert trainset.n_ratings == 3


def test_user_based_train_test_split():
    """Test user-based train/test split."""
    np.random.seed(42)
    
    # Create data with timestamps
    df = pd.DataFrame({
        "userId": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "movieId": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "rating": [4.0] * 10,
        "timestamp": list(range(5)) + list(range(5)),  # User 1: 0-4, User 2: 0-4
    })
    
    train, test = user_based_train_test_split(df, test_ratio=0.4)
    
    # Each user has 5 ratings, test_ratio=0.4 -> 2 test ratings per user
    assert len(test) == 4  # 2 users * 2 test ratings
    assert len(train) == 6  # 2 users * 3 train ratings
    # Test should have later timestamps (last 2 per user)
    for uid in [1, 2]:
        user_test = test[test["userId"] == uid]
        user_train = train[train["userId"] == uid]
        assert len(user_test) == 2
        assert len(user_train) == 3
        assert user_test["timestamp"].min() > user_train["timestamp"].max()


def test_time_based_train_test_split():
    """Test time-based train/test split."""
    df = pd.DataFrame({
        "userId": [1, 1, 2, 2],
        "movieId": [1, 2, 1, 2],
        "rating": [4.0] * 4,
        "timestamp": [1, 2, 3, 4],
    })
    
    train, test = time_based_train_test_split(df, test_ratio=0.5)
    
    assert len(train) == 2
    assert len(test) == 2
    assert train["timestamp"].max() < test["timestamp"].min()


def test_preprocessing_pipeline_runs():
    """Test that pipeline runs without error (smoke test)."""
    # This uses the actual data, just verify it completes
    result = run_preprocessing_pipeline(
        min_movie_ratings=10,
        min_user_ratings=10,
        test_ratio=0.2,
        split_method="user_based",
    )
    
    assert "movies_clean" in result
    assert "ratings_clean" in result
    assert "user_item_matrix" in result
    assert "mappings" in result
    assert "train_ratings" in result
    assert "test_ratings" in result
    
    # Check shapes
    assert result["ratings_clean"].shape[0] > 0
    assert result["train_ratings"].shape[0] > 0
    assert result["test_ratings"].shape[0] > 0
    assert result["user_item_matrix"].shape[0] == result["mappings"]["user_ids"].shape[0]
    assert result["user_item_matrix"].shape[1] == result["mappings"]["movie_ids"].shape[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])