"""Tests for the evaluation module (full-catalog ranking metrics)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation import (
    PopularityBaseline,
    coverage,
    evaluate_recommender,
    load_metrics,
    novelty,
    precision_recall_at_k,
    rank_candidates,
    save_metrics,
)
from recommender import load_models_and_data


@pytest.fixture(scope="module")
def eval_data():
    data, svd_algo = load_models_and_data()
    return {
        "trainset": data["train_data"].build_full_trainset(),
        "testset": data["test_data"].build_full_trainset().build_testset(),
        "svd": svd_algo,
    }


def test_rank_candidates_returns_raw_movie_ids(eval_data):
    """Ranked movie IDs must be raw MovieLens IDs present in the catalog."""
    trainset = eval_data["trainset"]
    items, scores = rank_candidates(eval_data["svd"], trainset, 1, k=10)

    assert len(items) == len(scores) == 10
    raw_movie_ids = {trainset.to_raw_iid(i) for i in range(trainset.n_items)}
    assert set(items).issubset(raw_movie_ids)
    assert scores == sorted(scores, reverse=True)


def test_rank_candidates_excludes_rated(eval_data):
    """Recommendations must not include movies the user already rated."""
    trainset = eval_data["trainset"]
    uid_inner = trainset.to_inner_uid(1)
    rated = {trainset.to_raw_iid(i) for i, _ in trainset.ur[uid_inner]}

    items, _ = rank_candidates(eval_data["svd"], trainset, 1, k=20)
    assert rated.isdisjoint(items)


def test_rank_candidates_scores_full_catalog(eval_data):
    """Scores must cover the whole catalog, not just held-out test items."""
    trainset = eval_data["trainset"]
    items, _ = rank_candidates(eval_data["svd"], trainset, 1, k=trainset.n_items)
    assert len(items) == trainset.n_items - len(trainset.ur[trainset.to_inner_uid(1)])


def test_rank_candidates_unknown_user(eval_data):
    items, scores = rank_candidates(eval_data["svd"], eval_data["trainset"], 99999, k=10)
    assert items == [] and scores == []


def test_precision_recall_at_k(eval_data):
    precision, recall, n_users = precision_recall_at_k(
        eval_data["svd"], eval_data["trainset"], eval_data["testset"], k=10
    )

    assert n_users > 0
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    # Full-catalog ranking is hard; precision should be small but non-zero.
    assert precision > 0


def test_coverage(eval_data):
    users = sorted({uid for uid, _, _ in eval_data["testset"]})
    cov = coverage(eval_data["svd"], eval_data["trainset"], users[:50], k=10)

    assert 0 < cov <= 1


def test_novelty(eval_data):
    users = sorted({uid for uid, _, _ in eval_data["testset"]})
    nov = novelty(eval_data["svd"], eval_data["trainset"], users[:50], k=10)

    assert 0 <= nov <= 1


def test_evaluate_recommender_keys(eval_data):
    metrics = evaluate_recommender(
        eval_data["svd"], eval_data["testset"], eval_data["trainset"], k=10, verbose=False
    )

    expected = {
        "rmse", "mae", "precision@10", "recall@10", "coverage@10",
        "novelty@10", "n_users", "k", "threshold",
    }
    assert set(metrics) == expected

    assert 0 < metrics["rmse"] < 1.5
    assert 0 < metrics["mae"] < 1.0
    assert 0 <= metrics["precision@10"] <= 1
    assert 0 <= metrics["recall@10"] <= 1
    assert 0 < metrics["coverage@10"] <= 1
    assert 0 <= metrics["novelty@10"] <= 1


def test_svd_beats_popularity_on_precision(eval_data):
    """SVD must rank better than a global popularity list."""
    svd_metrics = evaluate_recommender(
        eval_data["svd"], eval_data["testset"], eval_data["trainset"], k=10, verbose=False
    )
    pop_metrics = evaluate_recommender(
        PopularityBaseline(eval_data["trainset"]),
        eval_data["testset"],
        eval_data["trainset"],
        k=10,
        verbose=False,
    )

    assert svd_metrics["precision@10"] > pop_metrics["precision@10"]


def test_popularity_baseline_scores(eval_data):
    baseline = PopularityBaseline(eval_data["trainset"])
    assert baseline.fixed_scores.shape == (eval_data["trainset"].n_items,)
    assert len(baseline.test(eval_data["testset"])) == len(eval_data["testset"])


def test_save_and_load_metrics(tmp_path):
    payload = {"rmse": 0.85, "mae": 0.65, "precision@10": 0.01, "k": 10}
    path = save_metrics(payload, tmp_path / "metrics.json")

    assert path.exists()
    assert load_metrics(path) == payload
    assert load_metrics(tmp_path / "missing.json") is None
