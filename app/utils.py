"""Streamlit app utilities: poster display, formatting, caching."""
import streamlit as st
import requests
from functools import lru_cache
from typing import Optional, List, Dict
import pandas as pd

# OMDb API
OMDB_BASE_URL = "http://www.omdbapi.com/"
OMDB_API_KEY = st.secrets.get("OMDB_API_KEY", "")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_poster(imdb_id: str) -> Optional[str]:
    """
    Fetch movie poster URL from OMDb API with caching.
    
    Args:
        imdb_id: IMDb ID (e.g., 'tt0114709' or '0114709')
    
    Returns:
        Poster URL or None
    """
    if not OMDB_API_KEY:
        return None
    
    # Ensure proper IMDb ID format
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


def display_movie_card(movie: Dict, show_rating: bool = True) -> None:
    """
    Display a single movie card with poster, title, genres, year, rating.
    
    Args:
        movie: Dict with keys: title, year, genres, imdbId, estimated_rating/average_rating
        show_rating: Whether to show estimated/average rating
    """
    col1, col2 = st.columns([1, 3])
    
    with col1:
        poster_url = movie.get("poster_url")
        if poster_url:
            st.image(poster_url, width=120)
        else:
            # Placeholder
            st.markdown(
                f'<div style="width:120px;height:180px;background:#2c2c2c;display:flex;align-items:center;justify-content:center;border-radius:8px;color:#888;font-size:12px;">No Poster</div>',
                unsafe_allow_html=True
            )
    
    with col2:
        title = movie.get("title", "Unknown")
        year = movie.get("year")
        genres = movie.get("genres", [])
        
        st.markdown(f"### {title}" + (f" ({year})" if year else ""))
        
        if genres:
            genre_badges = " ".join([
                f'<span style="background:#3a3a3a;color:#ddd;padding:2px 8px;border-radius:12px;font-size:11px;margin-right:4px;">{g}</span>'
                for g in genres
            ])
            st.markdown(genre_badges, unsafe_allow_html=True)
        
        if show_rating:
            rating_key = "estimated_rating" if "estimated_rating" in movie else "average_rating"
            rating = movie.get(rating_key)
            if rating:
                stars = "⭐" * int(round(rating))
                st.markdown(f"**{rating:.1f}/5.0** {stars}")


def display_movie_grid(movies: List[Dict], cols: int = 4) -> None:
    """
    Display movies in a responsive grid.
    
    Args:
        movies: List of movie dicts
        cols: Number of columns
    """
    for i in range(0, len(movies), cols):
        row_movies = movies[i:i+cols]
        st_cols = st.columns(cols)
        
        for col, movie in zip(st_cols, row_movies):
            with col:
                poster_url = movie.get("poster_url")
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.markdown(
                        '<div style="height:250px;background:#2c2c2c;display:flex;align-items:center;justify-content:center;border-radius:8px;color:#888;">No Poster</div>',
                        unsafe_allow_html=True
                    )
                
                title = movie.get("title", "Unknown")
                year = movie.get("year")
                st.markdown(f"**{title}**" + (f" ({year})" if year else ""))
                
                genres = movie.get("genres", [])
                if genres:
                    genre_str = ", ".join(genres[:3])
                    if len(genres) > 3:
                        genre_str += f" +{len(genres)-3} more"
                    st.caption(genre_str)
                
                rating_key = "estimated_rating" if "estimated_rating" in movie else "average_rating"
                rating = movie.get(rating_key)
                if rating:
                    st.markdown(f"⭐ **{rating:.1f}**")


def format_genres(genres) -> str:
    """Format genres list/array as comma-separated string."""
    if hasattr(genres, 'tolist'):
        genres = genres.tolist()
    if isinstance(genres, list):
        return ", ".join(genres)
    return str(genres)


def show_loading_spinner(message: str = "Loading..."):
    """Show a loading spinner with message."""
    return st.spinner(message)


def show_error(message: str):
    """Show error message."""
    st.error(f"❌ {message}")


def show_success(message: str):
    """Show success message."""
    st.success(f"✅ {message}")


def show_info(message: str):
    """Show info message."""
    st.info(f"ℹ️ {message}")


@st.cache_data(ttl=3600, show_spinner=False)
def load_user_options() -> List[int]:
    """Load list of valid user IDs for dropdown."""
    from src.recommender import load_models_and_data
    try:
        data, _, _ = load_models_and_data()
        trainset = data["train_data"].build_full_trainset()
        mappings = data.get("mappings")
        if mappings is None:
            import joblib
            from pathlib import Path
            PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
            mappings = joblib.load(PROCESSED_DIR / "mappings.joblib")
        
        user_ids = sorted(mappings["user_ids"])
        return user_ids
    except Exception:
        return list(range(1, 611))


@st.cache_data(ttl=3600, show_spinner=False)
def get_all_movies_for_simulated_user() -> List[Dict]:
    """Get all movies with metadata for simulated user selection."""
    from src.preprocessing import load_preprocessed_data
    from pathlib import Path
    import pandas as pd
    
    try:
        data = load_preprocessed_data()
        movies = data["movies_clean"]
        links = pd.read_csv(Path(__file__).parent.parent / "data" / "raw" / "links.csv")
        
        meta = movies[["movieId", "clean_title", "genres_list", "year"]].copy()
        meta = meta.merge(links[["movieId", "imdbId", "tmdbId"]], on="movieId", how="left")
        
        results = []
        for _, row in meta.iterrows():
            genres = row["genres_list"]
            if hasattr(genres, 'tolist'):
                genres = genres.tolist()
            elif not isinstance(genres, list):
                genres = []
            
            imdb_id = row["imdbId"]
            poster = None
            if pd.notna(imdb_id):
                poster = fetch_poster(str(int(imdb_id)))
            
            results.append({
                "movieId": int(row["movieId"]),
                "title": row["clean_title"],
                "genres": genres,
                "year": int(row["year"]) if pd.notna(row["year"]) else None,
                "imdbId": f"tt{int(imdb_id):07d}" if pd.notna(imdb_id) else None,
                "poster_url": poster,
            })
        return results
    except Exception:
        return []


# Page config
def setup_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Movie Recommender",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .movie-card {
        background: #1e1e1e;
        border-radius: 12px;
        padding: 12px;
        transition: transform 0.2s;
    }
    .movie-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .stButton > button {
        width: 100%;
    }
    .genre-badge {
        display: inline-block;
        background: #3a3a3a;
        color: #ddd;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        margin: 2px 2px 2px 0;
    }
    .metric-card {
        background: #262730;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)