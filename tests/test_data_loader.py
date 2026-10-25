"""Tests for data_loader module."""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_raw_data, save_processed, load_processed


def test_load_raw_data_returns_dict():
    """load_raw_data should return a dict with 4 DataFrames."""
    data = load_raw_data()
    assert isinstance(data, dict)
    assert set(data.keys()) == {"movies", "ratings", "links", "tags"}


def test_load_raw_data_shapes():
    """Verify expected shapes for ml-latest-small."""
    data = load_raw_data()
    assert data["movies"].shape == (9742, 3)
    assert data["ratings"].shape == (100836, 4)
    assert data["links"].shape == (9742, 3)
    assert data["tags"].shape == (3683, 4)


def test_load_raw_data_columns():
    """Verify expected columns."""
    data = load_raw_data()
    assert list(data["movies"].columns) == ["movieId", "title", "genres"]
    assert list(data["ratings"].columns) == ["userId", "movieId", "rating", "timestamp"]
    assert list(data["links"].columns) == ["movieId", "imdbId", "tmdbId"]
    assert list(data["tags"].columns) == ["userId", "movieId", "tag", "timestamp"]


def test_load_raw_data_no_nulls_in_keys():
    """Key columns should not have nulls."""
    data = load_raw_data()
    assert data["movies"]["movieId"].notna().all()
    assert data["ratings"]["userId"].notna().all()
    assert data["ratings"]["movieId"].notna().all()
    assert data["ratings"]["rating"].notna().all()


def test_rating_scale():
    """Ratings should be in 1-5 range (with 0.5 increments)."""
    data = load_raw_data()
    ratings = data["ratings"]["rating"]
    assert ratings.min() >= 0.5
    assert ratings.max() <= 5.0
    # Check 0.5 increments
    unique_ratings = sorted(ratings.unique())
    expected = [i * 0.5 for i in range(1, 11)]  # 0.5, 1.0, ..., 5.0
    assert all(r in expected for r in unique_ratings)


def test_save_and_load_processed(tmp_path):
    """save_processed and load_processed roundtrip."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    save_processed("test_df", df)
    loaded = load_processed("test_df")
    pd.testing.assert_frame_equal(df, loaded)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])