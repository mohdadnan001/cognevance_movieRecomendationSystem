"""Tests for evaluation module."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation import (
    precision_recall_at_k,
    coverage,
    novelty,
    evaluate_recommender,
    popularity_baseline_recommendations,
)
from recommender import load_models_and_data


@pytest.fixture(scope="module")
def eval_data():
    """Load models and data for evaluation tests."""
    data, svd_algo, knn_algo = load_models_and_data()
    trainset = data["train_data"].build_full_trainset()
    testset = data["test_data"].build_full_trainset().build_testset()
    
    # Get predictions
    svd_predictions = svd_algo.test(testset)
    
    return {
        "trainset": trainset,
        "testset": testset,
        "svd_predictions": svd_predictions,
        "svd_algo": svd_algo,
    }


def test_precision_recall_at_k(eval_data):
    """Test precision@k and recall@k computation."""
    predictions = eval_data["svd_predictions"]
    
    precision, recall = precision_recall_at_k(predictions, k=10, threshold=4.0)
    
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert precision > 0.1  # Should be well above random


def test_coverage(eval_data):
    """Test catalog coverage computation."""
    predictions = eval_data["svd_predictions"]
    trainset = eval_data["trainset"]
    
    cov = coverage(predictions, trainset, k=10)
    
    assert 0 <= cov <= 1
    assert cov > 0.1  # Should recommend diverse items


def test_novelty(eval_data):
    """Test novelty computation."""
    predictions = eval_data["svd_predictions"]
    trainset = eval_data["trainset"]
    
    nov = novelty(predictions, trainset, k=10)
    
    assert 0 <= nov <= 1
    # Novelty should be reasonably high (not just recommending blockbusters)


def test_evaluate_recommender(eval_data):
    """Test comprehensive evaluation."""
    algo = eval_data["svd_algo"]
    testset = eval_data["testset"]
    trainset = eval_data["trainset"]
    
    metrics = evaluate_recommender(algo, testset, trainset, k=10, threshold=4.0)
    
    expected_keys = {"rmse", "mae", "precision@10", "recall@10", "coverage@10", "novelty@10"}
    assert set(metrics.keys()) == expected_keys
    
    assert 0 < metrics["rmse"] < 1.5
    assert 0 < metrics["mae"] < 1.0
    assert 0 <= metrics["precision@10"] <= 1
    assert 0 <= metrics["recall@10"] <= 1
    assert 0 <= metrics["coverage@10"] <= 1
    assert 0 <= metrics["novelty@10"] <= 1


def test_popularity_baseline(eval_data):
    """Test popularity baseline recommendations."""
    trainset = eval_data["trainset"]
    testset = eval_data["testset"]
    
    pop_predictions = popularity_baseline_recommendations(trainset, testset, k=10)
    
    assert len(pop_predictions) > 0
    assert all(hasattr(p, 'uid') and hasattr(p, 'iid') and hasattr(p, 'est') for p in pop_predictions)


def test_popularity_vs_svd_precision(eval_data):
    """Test that SVD beats popularity baseline on precision."""
    algo = eval_data["svd_algo"]
    trainset = eval_data["trainset"]
    testset = eval_data["testset"]
    
    svd_metrics = evaluate_recommender(algo, testset, trainset, k=10, threshold=4.0)
    
    pop_predictions = popularity_baseline_recommendations(trainset, testset, k=10)
    
    class PopBaseline:
        def test(self, ts):
            return pop_predictions
    
    pop_metrics = evaluate_recommender(PopBaseline(), testset, trainset, k=10, threshold=4.0)
    
    # SVD should significantly beat popularity on precision
    assert svd_metrics["precision@10"] > pop_metrics["precision@10"] * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])