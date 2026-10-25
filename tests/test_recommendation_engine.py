"""Tests for recommendation engine module."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recommendation_engine import (
    get_user_recommendations_with_metadata,
    get_cold_start_recommendations,
    get_user_rating_history,
)


def test_user_recommendations_svd():
    """Test SVD recommendations for known user."""
    recs = get_user_recommendations_with_metadata(1, "svd", top_n=10)
    
    assert isinstance(recs, list)
    assert len(recs) <= 10
    assert len(recs) > 0
    
    for r in recs:
        assert "movieId" in r
        assert "title" in r
        assert "genres" in r
        assert "year" in r
        assert "estimated_rating" in r
        assert isinstance(r["genres"], list)
        assert 1 <= r["estimated_rating"] <= 5


def test_user_recommendations_knn():
    """Test k-NN recommendations for known user."""
    recs = get_user_recommendations_with_metadata(1, "knn", top_n=10)
    
    assert isinstance(recs, list)
    assert len(recs) <= 10
    assert len(recs) > 0
    
    for r in recs:
        assert "estimated_rating" in r
        assert 1 <= r["estimated_rating"] <= 5


def test_user_recommendations_popularity():
    """Test popularity baseline recommendations."""
    recs = get_user_recommendations_with_metadata(1, "popularity", top_n=10)
    
    assert isinstance(recs, list)
    assert len(recs) <= 10
    assert len(recs) > 0
    
    for r in recs:
        assert "estimated_rating" in r  # Unified field name
        assert 1 <= r["estimated_rating"] <= 5


def test_invalid_model():
    """Test that invalid model name raises error."""
    try:
        get_user_recommendations_with_metadata(1, "invalid_model", top_n=5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_cold_start_recommendations():
    """Test cold start (popularity) recommendations."""
    recs = get_cold_start_recommendations(top_n=10)
    
    assert isinstance(recs, list)
    assert len(recs) <= 10
    assert len(recs) > 0
    
    for r in recs:
        assert "movieId" in r
        assert "title" in r
        assert "genres" in r
        assert "average_rating" in r
        assert isinstance(r["genres"], list)


def test_user_rating_history():
    """Test user rating history retrieval."""
    hist = get_user_rating_history(1, top_n=20)
    
    assert isinstance(hist, list)
    assert len(hist) <= 20
    assert len(hist) > 0
    
    for r in hist:
        assert "movieId" in r
        assert "title" in r
        assert "genres" in r
        assert "rating" in r
        assert isinstance(r["genres"], list)
        assert 0.5 <= r["rating"] <= 5.0
    
    # Check sorted by rating descending
    ratings = [r["rating"] for r in hist]
    assert ratings == sorted(ratings, reverse=True)


def test_unknown_user_recommendations():
    """Test recommendations for unknown user (cold start)."""
    recs = get_user_recommendations_with_metadata(99999, "svd", top_n=10)
    assert recs == []


def test_unknown_user_history():
    """Test rating history for unknown user."""
    hist = get_user_rating_history(99999, top_n=10)
    assert hist == []


def test_recommendations_have_metadata():
    """Test that recommendations include all required metadata fields."""
    recs = get_user_recommendations_with_metadata(1, "svd", top_n=5)
    
    required_fields = {"movieId", "title", "genres", "year", "imdbId", "tmdbId", "estimated_rating"}
    
    for r in recs:
        assert required_fields.issubset(r.keys())
        assert r["movieId"] > 0
        assert r["title"] is not None and len(r["title"]) > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])