"""Streamlit Movie Recommender App (SVD collaborative filtering)."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation import load_metrics
from src.recommendation_engine import (
    get_cold_start_recommendations,
    get_simulated_user_recommendations,
    get_user_rating_history,
    get_user_recommendations_with_metadata,
)
import utils
from utils import (
    display_movie_grid,
    load_movie_catalog,
    load_user_options,
    poster_for,
    setup_page,
    show_error,
    show_info,
    show_loading_spinner,
    show_success,
)

MODES = ["User ID", "Simulated User", "Popular (Cold Start)"]

setup_page()

st.session_state.setdefault("selected_movies", [])
st.session_state.setdefault("recommendations", [])
st.session_state.setdefault("current_mode", MODES[1])

# ---------------------------------------------------------------- sidebar
st.sidebar.title("🎬 Movie Recommender")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Recommendation Mode",
    MODES,
    index=MODES.index(st.session_state.current_mode),
)
st.session_state.current_mode = mode

top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)
show_metrics = st.sidebar.checkbox("Show Model Metrics", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** MovieLens ml-latest-small")
st.sidebar.markdown("**Algorithm:** SVD (matrix factorization)")

# ---------------------------------------------------------------- header
st.title("🎬 Movie Recommender System")
st.markdown("SVD collaborative filtering, ranked over the full movie catalog.")

if show_metrics and mode == "User ID":
    metrics = load_metrics()
    if metrics:
        k = metrics.get("k", 10)
        cols = st.columns(5)
        cols[0].metric("RMSE", f"{metrics['rmse']:.3f}", help="Lower is better")
        cols[1].metric("MAE", f"{metrics['mae']:.3f}", help="Lower is better")
        cols[2].metric(
            f"Precision@{k}",
            f"{metrics[f'precision@{k}']:.1%}",
            help="Of the recommended movies, the share the user rated 4+ in the held-out test set",
        )
        cols[3].metric(
            f"Recall@{k}",
            f"{metrics[f'recall@{k}']:.1%}",
            help="Of the user's 4+ rated test movies, the share that were recommended",
        )
        cols[4].metric(
            f"Coverage@{k}",
            f"{metrics[f'coverage@{k}']:.1%}",
            help="Share of the catalog that appears in some user's top recommendations",
        )
        st.caption(
            f"Full-catalog ranking over {metrics.get('n_users', 0)} users. "
            "Run `python -m src.evaluation` to refresh."
        )
    else:
        show_info("No saved metrics yet. Run `python -m src.evaluation` to generate them.")
    st.markdown("---")

# ================================================================ MODE 1
if mode == "User ID":
    st.subheader("👤 Get Recommendations for a User")

    col1, col2 = st.columns([3, 1])
    with col1:
        user_id = st.selectbox("Select User ID", load_user_options(), index=0)
    with col2:
        st.write("")
        st.write("")
        get_recs = st.button("🎯 Get Recommendations", type="primary", use_container_width=True)

    with st.expander("📜 View User's Rating History (Top 20)"):
        with show_loading_spinner("Loading user history..."):
            history = get_user_rating_history(user_id, top_n=20)

        if history:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Title": h["title"],
                            "Year": h["year"],
                            "Genres": ", ".join(h["genres"]),
                            "Rating": h["rating"],
                        }
                        for h in history
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            show_info("No rating history found for this user.")

    if get_recs:
        with show_loading_spinner(f"Generating {top_n} recommendations..."):
            try:
                st.session_state.recommendations = get_user_recommendations_with_metadata(
                    user_id, "svd", top_n=top_n, include_posters=True
                )
                show_success(f"Found {len(st.session_state.recommendations)} recommendations!")
            except Exception as exc:
                show_error(f"Error: {exc}")
                st.session_state.recommendations = []

    if st.session_state.recommendations:
        st.markdown(f"### 🎯 Top {top_n} Recommendations for User {user_id}")
        st.caption("SVD · full-catalog ranking · already-rated movies excluded")
        display_movie_grid(st.session_state.recommendations, cols=4)

# ================================================================ MODE 2
elif mode == "Simulated User":
    st.subheader("🎭 Simulated User Mode")
    st.markdown("Pick a few movies you like and we'll recommend similar ones.")

    with show_loading_spinner("Loading movie catalog..."):
        all_movies = load_movie_catalog()

    if not all_movies:
        show_error("Could not load movie catalog.")
        st.stop()

    st.markdown(f"**Catalog: {len(all_movies):,} movies** (filtered to ≥5 ratings)")

    col1, col2 = st.columns([3, 1])

    with col1:
        search = st.text_input("🔍 Search movies", placeholder="Type to filter...")
        filtered = all_movies
        if search:
            needle = search.lower()
            filtered = [m for m in all_movies if needle in m["title"].lower()]

        st.markdown(f"**Click movies to select/deselect:** ({len(filtered):,} shown, first 40)")
        shown = filtered[:40]

        st.markdown("---")
        if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
            with show_loading_spinner("Generating recommendations..."):
                try:
                    recs = get_simulated_user_recommendations(
                        st.session_state.selected_movies, top_n=top_n, include_posters=True
                    )
                    st.session_state.recommendations = recs
                    if recs:
                        show_success(f"Found {len(recs)} recommendations!")
                    else:
                        show_info("None of the selected movies are in the model's catalog.")
                except Exception as exc:
                    show_error(f"Error: {exc}")

        st.markdown("### Movie Picker")
        for start in range(0, len(shown), 4):
            columns = st.columns(4)
            for col, movie in zip(columns, shown[start : start + 4]):
                with col:
                    movie_id = movie["movieId"]
                    selected = movie_id in st.session_state.selected_movies

                    st.image(poster_for(movie), use_column_width=True)

                    label = (
                        f"{movie['title']} ({movie['year']})"
                        if movie.get("year")
                        else movie["title"]
                    )
                    checked = st.checkbox(
                        label,
                        value=selected,
                        key=f"movie_{movie_id}",
                        label_visibility="collapsed",
                    )
                    if checked and not selected:
                        st.session_state.selected_movies.append(movie_id)
                    elif not checked and selected:
                        st.session_state.selected_movies.remove(movie_id)

    with col2:
        st.markdown("### Your Selection")
        selected_ids = st.session_state.selected_movies

        if not selected_ids:
            st.info("Select movies from the catalog to get recommendations.")
        else:
            st.markdown(f"**{len(selected_ids)} movies selected**")
            by_id = {m["movieId"]: m for m in all_movies}
            for movie_id in selected_ids:
                st.caption(f"• {by_id[movie_id]['title']}")

            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.selected_movies = []
                st.session_state.recommendations = []
                st.rerun()

    if st.session_state.recommendations:
        st.markdown("### 🎯 Recommendations Based on Your Selection")
        display_movie_grid(st.session_state.recommendations, cols=4)

# ================================================================ MODE 3
else:
    st.subheader("🔥 Popular Movies (Cold Start)")
    st.markdown("Top-rated movies globally — a sensible start for brand-new users.")

    with show_loading_spinner("Loading popular movies..."):
        st.session_state.recommendations = get_cold_start_recommendations(top_n=top_n)

    if st.session_state.recommendations:
        show_success(f"Loaded {len(st.session_state.recommendations)} popular movies!")
        st.markdown(f"### 🔥 Top {top_n} Popular Movies")
        st.caption("Ranked by average rating (minimum 5 ratings)")
        display_movie_grid(st.session_state.recommendations, cols=4)
    else:
        show_error("No popular movies available — check the processed data.")

# ---------------------------------------------------------------- footer
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#888;font-size:12px;">'
    "Built with Python, scikit-surprise, Streamlit | Data: MovieLens ml-latest-small | "
    "Posters: OMDb API</div>",
    unsafe_allow_html=True,
)
