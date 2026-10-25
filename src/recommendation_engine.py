"""Recommendation engine with movie metadata enrichment and poster fetching."""
from pathlib import Path
import joblib
import requests
from functools import lru_cache
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np

from src.recommender import (
    load_models_and_data,
    get_top_n_recommendations,
    get_popularity_recommendations,
)
from src.preprocessing import load_preprocessed_data

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# OMDb API configuration
OMDB_BASE_URL = "http://www.omdbapi.com/"
OMDB_API_KEY = None  # Set via environment variable


def set_omdb_api_key(api_key: str):
    """Set OMDb API key."""
    global OMDB_API_KEY
    OMDB_API_KEY = api_key


@lru_cache(maxsize=1000)
def fetch_poster_omdb(imdb_id: str) -> Optional[str]:
    """
    Fetch movie poster URL from OMDb API.
    
    Args:
        imdb_id: IMDb ID (e.g., 'tt0114709')
    
    Returns:
        Poster URL or None if not found
    """
    if not OMDB_API_KEY:
        return None
    
    # Ensure IMDb ID format
    if not imdb_id.startswith("tt"):
        imdb_id = f"tt{imdb_id.zfill(7)}"
    
    try:
        params = {"apikey": OMDB_API_KEY, "i": imdb_id, "plot": "short"}
        response = requests.get(OMDB_BASE_URL, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get("Response") == "True" and data.get("Poster") != "N/A":
            return data["Poster"]
    except Exception:
        pass
    
    return None


def get_movie_metadata(movie_ids: List[int]) -> pd.DataFrame:
    """
    Get movie metadata for a list of movie IDs.
    
    Args:
        movie_ids: List of raw movie IDs
    
    Returns:
        DataFrame with movieId, title, genres, year, imdbId, tmdbId
    """
    # Load movies and links
    movies_clean = load_preprocessed_data()["movies_clean"]
    links = pd.read_csv(RAW_DIR / "links.csv")
    
    # Merge
    meta = movies_clean[["movieId", "clean_title", "genres_list", "year"]].copy()
    meta = meta.merge(links[["movieId", "imdbId", "tmdbId"]], on="movieId", how="left")
    
    # Filter to requested IDs
    meta = meta[meta["movieId"].isin(movie_ids)].copy()
    
    # Add poster URLs
    meta["poster_url"] = meta["imdbId"].apply(
        lambda x: fetch_poster_omdb(str(int(x))) if pd.notna(x) else None
    )
    
    return meta


def get_user_recommendations_with_metadata(
    user_id: int,
    model_name: str = "svd",
    top_n: int = 10,
    include_posters: bool = False,
) -> List[Dict]:
    """
    Get top-N recommendations for a user with full movie metadata.
    
    Args:
        user_id: Raw user ID (1-610)
        model_name: "svd", "knn", or "popularity"
        top_n: Number of recommendations
        include_posters: Whether to fetch posters from OMDb
    
    Returns:
        List of dicts with movie metadata and estimated rating
    """
    # Load models and data
    data, svd_algo, knn_algo = load_models_and_data()
    trainset = data["train_data"].build_full_trainset()
    movies_clean = data["movies_clean"]
    links = pd.read_csv(RAW_DIR / "links.csv")
    
    # Select model
    if model_name == "svd":
        algo = svd_algo
    elif model_name == "knn":
        algo = knn_algo
    elif model_name == "popularity":
        algo = None
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Get recommendations (inner IDs + estimated ratings)
    if model_name == "popularity":
        recs_inner = get_popularity_recommendations(trainset, n=top_n)
    else:
        recs_inner = get_top_n_recommendations(algo, user_id, trainset, n=top_n)
    
    if not recs_inner:
        return []
    
    # Map inner IDs to raw movie IDs
    idx_to_movie = data["mappings"]["idx_to_movie"] if "mappings" in data else None
    
    # If mappings not in data, load from joblib
    if idx_to_movie is None:
        mappings = joblib.load(PROCESSED_DIR / "mappings.joblib")
        idx_to_movie = mappings["idx_to_movie"]
    
    # Convert to raw movie IDs
    recs_raw = []
    for iid_inner, est_rating in recs_inner:
        movie_id = idx_to_movie.get(iid_inner)
        if movie_id:
            recs_raw.append((movie_id, float(est_rating)))
    
    # Get metadata
    movie_ids = [mid for mid, _ in recs_raw]
    meta = get_movie_metadata(movie_ids)
    
    # Build result
    results = []
    for movie_id, est_rating in recs_raw:
        movie_meta = meta[meta["movieId"] == movie_id]
        if len(movie_meta) == 0:
            continue
        
        row = movie_meta.iloc[0]
        # Handle genres_list (could be numpy array or list)
        genres = row["genres_list"]
        if hasattr(genres, 'tolist'):
            genres = genres.tolist()
        elif not isinstance(genres, list):
            genres = []
        
        result = {
            "movieId": int(movie_id),
            "title": row["clean_title"],
            "genres": genres,
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "imdbId": f"tt{int(row['imdbId']):07d}" if pd.notna(row["imdbId"]) else None,
            "tmdbId": int(row["tmdbId"]) if pd.notna(row["tmdbId"]) else None,
            "estimated_rating": round(est_rating, 2),
        }
        
        if include_posters and row.get("poster_url"):
            result["poster_url"] = row["poster_url"]
        
        results.append(result)
    
    return results


def get_cold_start_recommendations(top_n: int = 10, include_posters: bool = False) -> List[Dict]:
    """Get popularity-based recommendations for new/unknown users."""
    data, _, _ = load_models_and_data()
    trainset = data["train_data"].build_full_trainset()
    
    recs_inner = get_popularity_recommendations(trainset, n=top_n)
    
    # Map to raw IDs
    mappings = joblib.load(PROCESSED_DIR / "mappings.joblib")
    idx_to_movie = mappings["idx_to_movie"]
    
    recs_raw = []
    for iid_inner, avg_rating in recs_inner:
        movie_id = idx_to_movie.get(iid_inner)
        if movie_id:
            recs_raw.append((movie_id, float(avg_rating)))
    
    # Get metadata
    movie_ids = [mid for mid, _ in recs_raw]
    meta = get_movie_metadata(movie_ids)
    
    results = []
    for movie_id, avg_rating in recs_raw:
        movie_meta = meta[meta["movieId"] == movie_id]
        if len(movie_meta) == 0:
            continue
        
        row = movie_meta.iloc[0]
        # Handle genres_list (could be numpy array or list)
        genres = row["genres_list"]
        if hasattr(genres, 'tolist'):
            genres = genres.tolist()
        elif not isinstance(genres, list):
            genres = []
        
        result = {
            "movieId": int(movie_id),
            "title": row["clean_title"],
            "genres": genres,
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "imdbId": f"tt{int(row['imdbId']):07d}" if pd.notna(row["imdbId"]) else None,
            "tmdbId": int(row["tmdbId"]) if pd.notna(row["tmdbId"]) else None,
            "average_rating": round(avg_rating, 2),
        }
        
        if include_posters and row.get("poster_url"):
            result["poster_url"] = row["poster_url"]
        
        results.append(result)
    
    return results


def get_user_rating_history(user_id: int, top_n: int = 20) -> List[Dict]:
    """Get a user's highest-rated movies from training set."""
    data, _, _ = load_models_and_data()
    trainset = data["train_data"].build_full_trainset()
    movies_clean = data["movies_clean"]
    links = pd.read_csv(RAW_DIR / "links.csv")
    
    try:
        uid_inner = trainset.to_inner_uid(user_id)
    except ValueError:
        return []
    
    # Get user's ratings
    user_ratings = trainset.ur[uid_inner]
    if not user_ratings:
        return []
    
    # Sort by rating descending
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    top_ratings = user_ratings[:top_n]
    
    # Map to raw IDs
    mappings = joblib.load(PROCESSED_DIR / "mappings.joblib")
    idx_to_movie = mappings["idx_to_movie"]
    
    results = []
    for iid_inner, rating in top_ratings:
        movie_id = idx_to_movie.get(iid_inner)
        if not movie_id:
            continue
        
        movie_meta = movies_clean[movies_clean["movieId"] == movie_id]
        if len(movie_meta) == 0:
            continue
        
        row = movie_meta.iloc[0]
        link_row = links[links["movieId"] == movie_id]
        imdb_id = link_row.iloc[0]["imdbId"] if len(link_row) > 0 else None
        
        # Handle genres_list
        genres = row["genres_list"]
        if hasattr(genres, 'tolist'):
            genres = genres.tolist()
        elif not isinstance(genres, list):
            genres = []
        
        result = {
            "movieId": int(movie_id),
            "title": row["clean_title"],
            "genres": genres,
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "imdbId": f"tt{int(imdb_id):07d}" if pd.notna(imdb_id) else None,
            "rating": float(rating),
        }
        results.append(result)
    
    return results


if __name__ == "__main__":
    # Test recommendations
    print("Testing recommendations for user 1...")
    recs = get_user_recommendations_with_metadata(1, "svd", top_n=5)
    for r in recs:
        print(f"  {r['title']} ({r['year']}) - Est: {r['estimated_rating']} - Genres: {r['genres']}")
    
    print("\nTesting cold start recommendations...")
    cold = get_cold_start_recommendations(5)
    for r in cold:
        print(f"  {r['title']} ({r['year']}) - Avg: {r['average_rating']} - Genres: {r['genres']}")
    
    print("\nTesting user 1 rating history...")
    history = get_user_rating_history(1, top_n=5)
    for r in history:
        print(f"  {r['title']} ({r['year']}) - Rating: {r['rating']} - Genres: {r['genres']}")