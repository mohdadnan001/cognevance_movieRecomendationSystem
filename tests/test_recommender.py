"""Tests for recommender module."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recommender import (
    train_svd,
    train_knn,
    evaluate_model,
    get_top_n_recommendations,
    get_popularity_recommendations,
    save_model,
    load_model,
)
from preprocessing import load_preprocessed_data


@pytest.fixture(scope="module")
def trained_models():
    """Load pre-trained models and data."""
    data = load_preprocessed_data()
    trainset = data["train_data"].build_full_trainset()
    testset = data["test_data"].build_full_trainset().build_testset()
    
    # Load saved models
    svd_algo = load_model("algo_svd")
    knn_algo = load_model("algo_knn")
    
    return {
        "data": data,
        "trainset": trainset,
        "testset": testset,
        "svd": svd_algo,
        "knn": knn_algo,
    }


def test_svd_model_loaded(trained_models):
    """Test that SVD model loads correctly."""
    assert trained_models["svd"] is not None
    assert hasattr(trained_models["svd"], "predict")


def test_knn_model_loaded(trained_models):
    """Test that k-NN model loads correctly."""
    assert trained_models["knn"] is not None
    assert hasattr(trained_models["knn"], "predict")


def test_evaluate_model(trained_models):
    """Test model evaluation returns metrics."""
    svd = trained_models["svd"]
    testset = trained_models["testset"]
    
    results = evaluate_model(svd, testset, verbose=False)
    
    assert "rmse" in results
    assert "mae" in results
    assert "predictions" in results
    assert results["rmse"] < 1.0  # Should be well under 1.0
    assert results["mae"] < 0.8
    assert len(results["predictions"]) == len(testset)


def test_get_top_n_recommendations(trained_models):
    """Test top-N recommendation generation."""
    svd = trained_models["svd"]
    trainset = trained_models["trainset"]
    
    # Test for a known user
    recs = get_top_n_recommendations(svd, 1, trainset, n=10)
    
    assert isinstance(recs, list)
    assert len(recs) <= 10
    assert all(isinstance(item, tuple) and len(item) == 2 for item in recs)
    # Check sorted by estimated rating descending
    est_ratings = [est for _, est in recs]
    assert est_ratings == sorted(est_ratings, reverse=True)


def test_get_top_n_recommendations_cold_start(trained_models):
    """Test recommendations for unknown user (cold start)."""
    svd = trained_models["svd"]
    trainset = trained_models["trainset"]
    
    # User ID that doesn't exist
    recs = get_top_n_recommendations(svd, 99999, trainset, n=10)
    assert recs == []


def test_get_popularity_recommendations(trained_models):
    """Test popularity-based recommendations."""
    trainset = trained_models["trainset"]
    
    recs = get_popularity_recommendations(trainset, n=10)
    
    assert isinstance(recs, list)
    assert len(recs) <= 10
    assert all(isinstance(item, tuple) and len(item) == 2 for item in recs)
    # Check sorted by average rating descending
    avg_ratings = [avg for _, avg in recs]
    assert avg_ratings == sorted(avg_ratings, reverse=True)


def test_model_comparison(trained_models):
    """Test that SVD outperforms k-NN on RMSE."""
    svd = trained_models["svd"]
    knn = trained_models["knn"]
    testset = trained_models["testset"]
    
    svd_results = evaluate_model(svd, testset, verbose=False)
    knn_results = evaluate_model(knn, testset, verbose=False)
    
    # SVD should have better (lower) RMSE and MAE
    assert svd_results["rmse"] < knn_results["rmse"]
    assert svd_results["mae"] < knn_results["mae"]


def test_recommendations_exclude_rated(trained_models):
    """Test that recommendations exclude already-rated items."""
    svd = trained_models["svd"]
    trainset = trained_models["trainset"]
    
    # Get user's rated items in trainset
    uid_inner = trainset.to_inner_uid(1)
    rated_items = set(iid for iid, _ in trainset.ur[uid_inner])
    
    recs = get_top_n_recommendations(svd, 1, trainset, n=20, exclude_rated=True)
    rec_items = set(iid for iid, _ in recs)
    
    # No overlap between rated and recommended
    assert len(rated_items & rec_items) == 0


def test_save_load_model(tmp_path, trained_models):
    """Test model save/load roundtrip."""
    svd = trained_models["svd"]
    
    # Save to temp location
    import joblib
    test_path = tmp_path / "test_model.pkl"
    joblib.dump(svd, test_path)
    
    # Load back
    loaded = joblib.load(test_path)
    
    assert loaded is not None
    assert hasattr(loaded, "predict")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])