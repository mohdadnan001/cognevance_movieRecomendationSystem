"""Tests for the recommendation engine module."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recommendation_engine import (
    get_cold_start_recommendations,
    get_simulated_user_recommendations,
    get_user_rating_history,
    get_user_recommendations_with_metadata,
)


def test_user_recommendations_svd():
    recs = get_user_recommendations_with_metadata(1, "svd", top_n=10)

    assert 0 < len(recs) <= 10
    for r in recs:
        assert {"movieId", "title", "genres", "year", "estimated_rating"} <= r.keys()
        assert isinstance(r["genres"], list)
        assert 0.5 <= r["estimated_rating"] <= 5.0


def test_recommendations_are_personalised():
    """Two users with different histories should not get identical lists."""
    recs_1 = get_user_recommendations_with_metadata(1, "svd", top_n=10)
    recs_2 = get_user_recommendations_with_metadata(1, "svd", top_n=10)
    assert [r["movieId"] for r in recs_1] == [r["movieId"] for r in recs_2]

    recs_other = get_user_recommendations_with_metadata(42, "svd", top_n=10)
    assert [r["movieId"] for r in recs_1] != [r["movieId"] for r in recs_other]


def test_user_recommendations_popularity():
    """model_name='popularity' is the documented cold-start path."""
    recs = get_user_recommendations_with_metadata(1, "popularity", top_n=10)

    assert 0 < len(recs) <= 10
    for r in recs:
        assert 0.5 <= r["estimated_rating"] <= 5.0


def test_invalid_model():
    with pytest.raises(ValueError):
        get_user_recommendations_with_metadata(1, "knn", top_n=5)


def test_cold_start_recommendations():
    recs = get_cold_start_recommendations(top_n=10)

    assert 0 < len(recs) <= 10
    for r in recs:
        assert {"movieId", "title", "genres", "average_rating"} <= r.keys()
        assert isinstance(r["genres"], list)


def test_user_rating_history():
    hist = get_user_rating_history(1, top_n=20)

    assert 0 < len(hist) <= 20
    for r in hist:
        assert {"movieId", "title", "genres", "rating"} <= r.keys()
        assert isinstance(r["genres"], list)
        assert 0.5 <= r["rating"] <= 5.0

    ratings = [r["rating"] for r in hist]
    assert ratings == sorted(ratings, reverse=True)


def test_simulated_user_recommendations():
    """Selections drive the recommendations and are not recommended back."""
    selected = [1, 2, 3]  # Toy Story, Jumanji, Grumpier Old Men
    recs = get_simulated_user_recommendations(selected, top_n=10)

    assert 0 < len(recs) <= 10
    assert set(selected).isdisjoint({r["movieId"] for r in recs})
    assert [r["estimated_rating"] for r in recs] == sorted(
        (r["estimated_rating"] for r in recs), reverse=True
    )


def test_simulated_user_unknown_movies():
    assert get_simulated_user_recommendations([], top_n=10) == []
    assert get_simulated_user_recommendations([10**9], top_n=10) == []


def test_unknown_user_recommendations():
    assert get_user_recommendations_with_metadata(99999, "svd", top_n=10) == []


def test_unknown_user_history():
    assert get_user_rating_history(99999, top_n=10) == []


def test_recommendations_have_metadata():
    required = {"movieId", "title", "genres", "year", "imdbId", "tmdbId", "estimated_rating"}

    for r in get_user_recommendations_with_metadata(1, "svd", top_n=5):
        assert required.issubset(r.keys())
        assert r["movieId"] > 0
        assert r["title"]
