"""Tests for the SVD recommender module."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import load_preprocessed_data
from recommender import (
    evaluate_model,
    get_popularity_recommendations,
    get_top_n_recommendations,
    load_model,
    save_model,
)


@pytest.fixture(scope="module")
def trained_model():
    data = load_preprocessed_data()
    return {
        "data": data,
        "trainset": data["train_data"].build_full_trainset(),
        "testset": data["test_data"].build_full_trainset().build_testset(),
        "svd": load_model("algo_svd"),
    }


def test_svd_model_loaded(trained_model):
    assert trained_model["svd"] is not None
    assert hasattr(trained_model["svd"], "predict")


def test_evaluate_model(trained_model):
    results = evaluate_model(trained_model["svd"], trained_model["testset"], verbose=False)

    assert results["rmse"] < 1.0
    assert results["mae"] < 0.8
    assert len(results["predictions"]) == len(trained_model["testset"])


def test_get_top_n_recommendations(trained_model):
    recs = get_top_n_recommendations(trained_model["svd"], 1, trained_model["trainset"], n=10)

    assert len(recs) == 10
    assert all(len(item) == 2 for item in recs)
    scores = [score for _, score in recs]
    assert scores == sorted(scores, reverse=True)


def test_recommendations_use_raw_movie_ids(trained_model):
    """Ranked IDs must map to known movies and match algo.predict's score."""
    trainset = trained_model["trainset"]
    recs = get_top_n_recommendations(trained_model["svd"], 1, trainset, n=5)

    for movie_id, score in recs:
        predicted = trained_model["svd"].predict(1, movie_id, verbose=False).est
        assert score == pytest.approx(predicted, abs=1e-6)


def test_get_top_n_recommendations_cold_start(trained_model):
    recs = get_top_n_recommendations(trained_model["svd"], 99999, trained_model["trainset"], n=10)
    assert recs == []


def test_recommendations_exclude_rated(trained_model):
    trainset = trained_model["trainset"]
    uid_inner = trainset.to_inner_uid(1)
    rated = {trainset.to_raw_iid(i) for i, _ in trainset.ur[uid_inner]}

    recs = get_top_n_recommendations(trained_model["svd"], 1, trainset, n=20)
    assert rated.isdisjoint({movie_id for movie_id, _ in recs})


def test_get_popularity_recommendations(trained_model):
    trainset = trained_model["trainset"]
    recs = get_popularity_recommendations(trainset, n=10)

    assert len(recs) == 10
    averages = [avg for _, avg in recs]
    assert averages == sorted(averages, reverse=True)
    assert all(1.0 <= avg <= 5.0 for avg in averages)


def test_save_load_model(tmp_path, trained_model):
    import joblib

    path = tmp_path / "svd.pkl"
    joblib.dump(trained_model["svd"], path)

    loaded = joblib.load(path)
    assert hasattr(loaded, "predict")


def test_project_is_svd_only():
    """k-NN was removed; ensure the public API does not expose it again."""
    import recommender

    assert not hasattr(recommender, "train_knn")
