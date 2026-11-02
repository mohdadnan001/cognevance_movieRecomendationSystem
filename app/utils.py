"""Streamlit app utilities: page setup, posters, movie grid, caching.

Posters are fetched lazily, only for movies actually rendered. Fetching the whole
catalog up front issued thousands of OMDb requests and stalled the first page load.
"""
import os
import sys
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OMDB_BASE_URL = "http://www.omdbapi.com/"

load_dotenv(ROOT_DIR / ".env", override=False)


def _get_omdb_api_key() -> str:
    """Read the OMDb key from Streamlit secrets, falling back to the environment."""
    try:
        key = st.secrets.get("OMDB_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("OMDB_API_KEY", "")


OMDB_API_KEY = _get_omdb_api_key()


def _placeholder_poster(title: str = "Movie", year: int = None) -> str:
    """Generate a fallback placeholder poster URL."""
    text = f"{title} {year}".strip() if year else title
    encoded = urllib.parse.quote(text)
    return f"https://placehold.co/300x450/1a1a1a/FF6B6B?text={encoded}"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_poster(imdb_id: str, title: str = "", year: int = None) -> str:
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


@st.cache_data(ttl=3600, show_spinner=False)
def load_user_options() -> List[int]:
    """Valid user IDs for the dropdown."""
    from src.recommendation_engine import _load

    try:
        data, _, _ = _load()
        return sorted(int(u) for u in data["mappings"]["user_ids"])
    except Exception:
        return list(range(1, 611))


@st.cache_data(ttl=3600, show_spinner=False)
def load_movie_catalog() -> List[Dict]:
    """Movie catalog for the simulated-user picker (no posters; those load lazily)."""
    from src.recommendation_engine import _movie_catalog

    try:
        catalog = _movie_catalog()
    except Exception:
        return []

    results = []
    for _, row in catalog.iterrows():
        genres = row["genres_list"]
        if hasattr(genres, "tolist"):
            genres = genres.tolist()
        elif not isinstance(genres, list):
            genres = []

        imdb_id = row["imdbId"]
        results.append(
            {
                "movieId": int(row["movieId"]),
                "title": row["clean_title"],
                "genres": genres,
                "year": int(row["year"]) if pd.notna(row["year"]) else None,
                "imdbId": f"tt{int(imdb_id):07d}" if pd.notna(imdb_id) else None,
            }
        )
    return sorted(results, key=lambda m: m["title"])


def poster_for(movie: Dict) -> str:
    """Poster URL for a movie dict, using any pre-supplied URL first."""
    if movie.get("poster_url"):
        return movie["poster_url"]
    return fetch_poster(
        movie.get("imdbId") or "",
        title=movie.get("title", ""),
        year=movie.get("year"),
    )


def display_movie_grid(movies: List[Dict], cols: int = 4, show_rating: bool = True) -> None:
    """Display movies in a responsive grid, fetching each poster as it is drawn."""
    for start in range(0, len(movies), cols):
        row_movies = movies[start : start + cols]
        columns = st.columns(cols)

        for col, movie in zip(columns, row_movies):
            with col:
                url = poster_for(movie)
                if url:
                    st.image(url, use_column_width=True)
                else:
                    st.markdown(
                        '<div style="height:250px;background:#2c2c2c;display:flex;'
                        'align-items:center;justify-content:center;border-radius:8px;'
                        'color:#888;">No Poster</div>',
                        unsafe_allow_html=True,
                    )

                title = movie.get("title", "Unknown")
                year = movie.get("year")
                st.markdown(f"**{title}**" + (f" ({year})" if year else ""))

                genres = movie.get("genres", [])
                if genres:
                    genre_str = ", ".join(genres[:3])
                    if len(genres) > 3:
                        genre_str += f" +{len(genres) - 3} more"
                    st.caption(genre_str)

                if show_rating:
                    rating = movie.get("estimated_rating", movie.get("average_rating"))
                    if rating is not None:
                        st.markdown(f"⭐ **{rating:.1f}**")


def format_genres(genres) -> str:
    """Format a genres list/array as a comma-separated string."""
    if hasattr(genres, "tolist"):
        genres = genres.tolist()
    return ", ".join(genres) if isinstance(genres, list) else str(genres)


def show_loading_spinner(message: str = "Loading..."):
    """Show a loading spinner with message."""
    return st.spinner(message)


def show_error(message: str) -> None:
    """Show an error message."""
    st.error(f"❌ {message}")


def show_success(message: str) -> None:
    """Show a success message."""
    st.success(f"✅ {message}")


def show_info(message: str) -> None:
    """Show an info message."""
    st.info(f"ℹ️ {message}")


def setup_page() -> None:
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="Movie Recommender",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
        .stButton > button { width: 100%; }
        </style>
        """,
        unsafe_allow_html=True,
    )
