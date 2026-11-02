"""Recommendation engine: SVD recommendations enriched with movie metadata.

``src.recommender`` returns raw MovieLens movie IDs for every recommendation, so
the metadata layer only has to join against ``movies.csv`` / ``links.csv`` - no
inner-ID mapping is involved (mapping Surprise inner IDs through a separately
built ``mappings.joblib`` produced wrong movies).
"""
import os
import urllib.parse
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.recommender import (
    get_popularity_recommendations,
    get_top_n_recommendations,
    load_models_and_data,
)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# OMDb API configuration
OMDB_BASE_URL = "http://www.omdbapi.com/"

_load_dotenv = load_dotenv
_load_dotenv(Path(__file__).parent.parent / ".env", override=False)

OMDB_API_KEY = os.getenv("OMDB_API_KEY", "")


def set_omdb_api_key(api_key: str) -> None:
    """Set OMDb API key."""
    global OMDB_API_KEY
    OMDB_API_KEY = api_key


def _placeholder_poster(title: str = "Movie", year: int = None) -> str:
    """Generate a fallback placeholder poster URL."""
    text = f"{title} {year}".strip() if year else title
    encoded = urllib.parse.quote(text)
    return f"https://placehold.co/300x450/1a1a1a/FF6B6B?text={encoded}"


@lru_cache(maxsize=2000)
def fetch_poster_omdb(imdb_id: str, title: str = "", year: int = None) -> str:
    """Fetch a poster URL from OMDb, always returning a usable URL."""
    if not OMDB_API_KEY or not imdb_id:
        return _placeholder_poster(title, year)

    if not imdb_id.startswith("tt"):
        imdb_id = f"tt{imdb_id.zfill(7)}"

    try:
        response = requests.get(
            OMDB_BASE_URL,
            params={"apikey": OMDB_API_KEY, "i": imdb_id, "plot": "short"},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("Response") == "True" and data.get("Poster") != "N/A":
            return data["Poster"]
    except Exception:
        pass

    return _placeholder_poster(title, year)


@lru_cache(maxsize=1)
def _load():
    """Load preprocessed data, the SVD model and its trainset once per process."""
    data, svd_algo = load_models_and_data()
    return data, svd_algo, data["train_data"].build_full_trainset()


@lru_cache(maxsize=1)
def _movie_catalog() -> pd.DataFrame:
    """Cleaned movies joined with the IMDb/TMDB links, indexed by ``movieId``."""
    data, _, _ = _load()
    movies = data["movies_clean"][["movieId", "clean_title", "genres_list", "year"]]
    links = pd.read_csv(RAW_DIR / "links.csv")[["movieId", "imdbId", "tmdbId"]]
    catalog = movies.merge(links, on="movieId", how="left")
    return catalog.set_index("movieId", drop=False)


def _genres(row) -> List[str]:
    """Normalise a ``genres_list`` cell (parquet may return numpy arrays)."""
    genres = row["genres_list"]
    if hasattr(genres, "tolist"):
        genres = genres.tolist()
    return genres if isinstance(genres, list) else []


def _build_result(row, score: float, score_key: str, include_posters: bool) -> Dict:
    """Build the metadata dict the Streamlit app renders."""
    imdb_id = row["imdbId"]
    year = row["year"]

    result = {
        "movieId": int(row["movieId"]),
        "title": row["clean_title"],
        "genres": _genres(row),
        "year": int(year) if pd.notna(year) else None,
        "imdbId": f"tt{int(imdb_id):07d}" if pd.notna(imdb_id) else None,
        "tmdbId": int(row["tmdbId"]) if pd.notna(row["tmdbId"]) else None,
    }

    if score_key == "estimated_rating":
        result["estimated_rating"] = round(score, 2)
    else:
        result["average_rating"] = round(score, 2)

    if include_posters:
        result["poster_url"] = fetch_poster_omdb(
            str(int(imdb_id)) if pd.notna(imdb_id) else "",
            title=row["clean_title"],
            year=result["year"],
        )

    return result


def _enrich(ranked, score_key: str, include_posters: bool) -> List[Dict]:
    """Join ranked ``(movieId, score)`` pairs with movie metadata."""
    if not ranked:
        return []

    catalog = _movie_catalog()
    results = []
    for movie_id, score in ranked:
        if movie_id not in catalog.index:
            continue
        results.append(_build_result(catalog.loc[movie_id], score, score_key, include_posters))
    return results


def get_user_recommendations_with_metadata(
    user_id: int,
    model_name: str = "svd",
    top_n: int = 10,
    include_posters: bool = False,
) -> List[Dict]:
    """
    Get top-N SVD recommendations for a user, with full movie metadata.

    Args:
        user_id: Raw MovieLens user ID.
        model_name: ``"svd"`` or ``"popularity"`` (cold-start fallback).
        top_n: Number of recommendations.
        include_posters: Whether to fetch posters from OMDb.

    Returns:
        List of metadata dicts. Empty for an unknown user under ``"svd"``.
    """
    _, svd_algo, trainset = _load()

    if model_name == "popularity":
        ranked = get_popularity_recommendations(trainset, n=top_n)
        return _enrich(ranked, "estimated_rating", include_posters)

    if model_name != "svd":
        raise ValueError(f"Unknown model: {model_name}")

    ranked = get_top_n_recommendations(svd_algo, user_id, trainset, n=top_n)
    return _enrich(ranked, "estimated_rating", include_posters)


def get_simulated_user_recommendations(
    movie_ids: List[int],
    top_n: int = 10,
    include_posters: bool = False,
) -> List[Dict]:
    """Recommend from a set of movies the visitor picked themselves.

    SVD has no user factor for an unknown visitor, so we build one by averaging
    the item factors of the chosen movies and score the catalog with it. Items
    the visitor already picked are excluded.
    """
    if not movie_ids:
        return []

    _, svd_algo, trainset = _load()

    inner_ids = []
    for movie_id in movie_ids:
        try:
            inner_ids.append(trainset.to_inner_iid(movie_id))
        except ValueError:
            continue  # Movie filtered out of the trainset

    if not inner_ids:
        return []

    profile = svd_algo.qi[inner_ids].mean(axis=0)
    scores = trainset.global_mean + svd_algo.bi + svd_algo.qi @ profile
    lo, hi = trainset.rating_scale
    scores = np.clip(scores, lo, hi)

    # Drop the visitor's own picks, and movies too rare to score meaningfully.
    for inner_id in inner_ids:
        scores[inner_id] = -np.inf

    top_n = min(top_n, int(np.isfinite(scores).sum()))
    if top_n <= 0:
        return []

    top = np.argpartition(-scores, top_n - 1)[:top_n]
    top = top[np.argsort(-scores[top])]

    ranked = [(trainset.to_raw_iid(i), float(scores[i])) for i in top]
    return _enrich(ranked, "estimated_rating", include_posters)


def get_cold_start_recommendations(top_n: int = 10, include_posters: bool = False) -> List[Dict]:
    """Popularity-based recommendations for new/unknown users."""
    _, _, trainset = _load()

    ranked = get_popularity_recommendations(trainset, n=top_n)
    return _enrich(ranked, "average_rating", include_posters)


def get_user_rating_history(user_id: int, top_n: int = 20) -> List[Dict]:
    """Get a user's highest-rated movies from the training set."""
    _, _, trainset = _load()

    try:
        uid_inner = trainset.to_inner_uid(user_id)
    except ValueError:
        return []

    user_ratings = sorted(trainset.ur[uid_inner], key=lambda x: x[1], reverse=True)[:top_n]

    catalog = _movie_catalog()
    results = []
    for iid_inner, rating in user_ratings:
        movie_id = trainset.to_raw_iid(iid_inner)
        if movie_id not in catalog.index:
            continue
        result = _build_result(catalog.loc[movie_id], float(rating), "estimated_rating", False)
        result["rating"] = result.pop("estimated_rating")
        results.append(result)

    return results


if __name__ == "__main__":
    print("User 1 (SVD):")
    for r in get_user_recommendations_with_metadata(1, "svd", top_n=5):
        print(f"  {r['title']} ({r['year']}) - Est: {r['estimated_rating']} - {r['genres']}")

    print("\nCold start:")
    for r in get_cold_start_recommendations(5):
        print(f"  {r['title']} ({r['year']}) - Avg: {r['average_rating']} - {r['genres']}")

    print("\nUser 1 history:")
    for r in get_user_rating_history(1, top_n=5):
        print(f"  {r['title']} ({r['year']}) - Rating: {r['rating']} - {r['genres']}")
