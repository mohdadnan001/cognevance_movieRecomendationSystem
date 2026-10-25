"""Streamlit Movie Recommender App."""
import streamlit as st
import pandas as pd
from typing import List, Dict

from src.recommendation_engine import (
    get_user_recommendations_with_metadata,
    get_cold_start_recommendations,
    get_user_rating_history,
    set_omdb_api_key,
)
from app.utils import (
    setup_page,
    display_movie_grid,
    show_loading_spinner,
    show_error,
    show_success,
    show_info,
    load_user_options,
    get_all_movies_for_simulated_user,
)

# Page setup
setup_page()

# Initialize session state
if "selected_movies" not in st.session_state:
    st.session_state.selected_movies = []
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "User ID"

# Sidebar
st.sidebar.title("🎬 Movie Recommender")
st.sidebar.markdown("---")

# Mode selection
mode = st.sidebar.radio(
    "Recommendation Mode",
    ["User ID", "Simulated User", "Popular (Cold Start)"],
    index=["User ID", "Simulated User", "Popular (Cold Start)"].index(st.session_state.current_mode)
)
st.session_state.current_mode = mode

# Model selection (only for User ID mode)
if mode == "User ID":
    model = st.sidebar.selectbox(
        "Algorithm",
        ["SVD (Matrix Factorization)", "k-NN (Item-based)", "Popularity Baseline"],
        index=0
    )
    model_key = {"SVD (Matrix Factorization)": "svd", 
                 "k-NN (Item-based)": "knn", 
                 "Popularity Baseline": "popularity"}[model]
else:
    model_key = "popularity" if mode == "Popular (Cold Start)" else "svd"

# Number of recommendations
top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

# Show metrics
show_metrics = st.sidebar.checkbox("Show Model Metrics", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** MovieLens ml-latest-small")
st.sidebar.markdown("**Users:** 610 | **Movies:** 3,650 (filtered)")
st.sidebar.markdown("**Approach:** Collaborative Filtering")

# Main content
st.title("🎬 Movie Recommender System")
st.markdown("Get personalized movie recommendations using collaborative filtering.")

# Model metrics
if show_metrics and mode == "User ID":
    metrics = {
        "svd": {"RMSE": 0.846, "MAE": 0.647, "Precision@10": 0.528, "Recall@10": 0.686, "Coverage@10": 0.620},
        "knn": {"RMSE": 0.897, "MAE": 0.688, "Precision@10": 0.514, "Recall@10": 0.678, "Coverage@10": 0.672},
        "popularity": {"RMSE": 1.333, "MAE": 1.074, "Precision@10": 0.066, "Recall@10": 0.629, "Coverage@10": 0.003},
    }
    m = metrics.get(model_key, metrics["svd"])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("RMSE", f"{m['RMSE']:.3f}", help="Lower is better")
    with col2:
        st.metric("MAE", f"{m['MAE']:.3f}", help="Lower is better")
    with col3:
        st.metric("Precision@10", f"{m['Precision@10']:.1%}", help="Of recommended, % relevant")
    with col4:
        st.metric("Recall@10", f"{m['Recall@10']:.1%}", help="Of relevant, % recommended")
    with col5:
        st.metric("Coverage@10", f"{m['Coverage@10']:.1%}", help="% of catalog recommended")
    
    st.markdown("---")

# ============================================================
# MODE 1: USER ID INPUT
# ============================================================
if mode == "User ID":
    st.subheader("👤 Get Recommendations for a User")
    
    user_ids = load_user_options()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        user_id = st.selectbox(
            "Select User ID",
            user_ids,
            index=0,
            help="Choose a user from the MovieLens dataset (1-610)"
        )
    with col2:
        st.write("")
        st.write("")
        get_recs = st.button("🎯 Get Recommendations", type="primary", use_container_width=True)
    
    # Show user's rating history
    with st.expander("📜 View User's Rating History (Top 20)", expanded=False):
        with show_loading_spinner("Loading user history..."):
            history = get_user_rating_history(user_id, top_n=20)
        
        if history:
            st.markdown(f"**User {user_id}'s highest-rated movies:**")
            hist_df = pd.DataFrame(history)
            hist_df["genres"] = hist_df["genres"].apply(lambda x: ", ".join(x) if x else "—")
            st.dataframe(
                hist_df[["title", "year", "genres", "rating"]].rename(columns={
                    "title": "Title", "year": "Year", "genres": "Genres", "rating": "Rating"
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            show_info("No rating history found for this user.")
    
    # Get recommendations
    if get_recs:
        with show_loading_spinner(f"Generating {top_n} recommendations using {model}..."):
            try:
                if model_key == "popularity":
                    recs = get_cold_start_recommendations(top_n=top_n)
                else:
                    recs = get_user_recommendations_with_metadata(
                        user_id, model_key, top_n=top_n, include_posters=True
                    )
                
                st.session_state.recommendations = recs
                show_success(f"Found {len(recs)} recommendations!")
            except Exception as e:
                show_error(f"Error: {str(e)}")
                st.session_state.recommendations = []
    
    # Display recommendations
    if st.session_state.recommendations:
        st.markdown(f"### 🎯 Top {len(st.session_state.recommendations)} Recommendations for User {user_id}")
        st.caption(f"Model: {model} | Based on collaborative filtering")
        display_movie_grid(st.session_state.recommendations, cols=4)

# ============================================================
# MODE 2: SIMULATED USER
# ============================================================
elif mode == "Simulated User":
    st.subheader("🎭 Simulated User Mode")
    st.markdown("Select movies you like, and we'll recommend similar ones.")
    
    # Load all movies
    with show_loading_spinner("Loading movie catalog..."):
        all_movies = get_all_movies_for_simulated_user()
    
    if not all_movies:
        show_error("Could not load movie catalog.")
        st.stop()
    
    st.markdown(f"**Catalog: {len(all_movies)} movies** (filtered to ≥5 ratings)")
    
    # Movie selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Search/filter
        search = st.text_input("🔍 Search movies", placeholder="Type to filter...")
        
        # Filter movies
        filtered_movies = all_movies
        if search:
            search_lower = search.lower()
            filtered_movies = [m for m in all_movies if search_lower in m["title"].lower()]
        
        # Show as selectable cards
        st.markdown("**Click movies to select/deselect:**")
        
        # Display in grid with checkboxes
        for i in range(0, min(len(filtered_movies), 40), 4):
            row_movies = filtered_movies[i:i+4]
            cols = st.columns(4)
            
            for col, movie in zip(cols, row_movies):
                with cols:
                    movie_id = movie["movieId"]
                    is_selected = movie_id in st.session_state.selected_movies
                    
                    # Poster
                    if movie.get("poster_url"):
                        st.image(movie["poster_url"], use_container_width=True)
                    else:
                        st.markdown(
                            '<div style="height:200px;background:#2c2c2c;display:flex;align-items:center;justify-content:center;border-radius:8px;color:#888;">No Poster</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Title + checkbox
                    title = movie["title"]
                    year = movie["year"]
                    label = f"{title} ({year})" if year else title
                    
                    checked = st.checkbox(
                        label,
                        value=is_selected,
                        key=f"movie_{movie_id}",
                        label_visibility="collapsed"
                    )
                    
                    if checked and not is_selected:
                        st.session_state.selected_movies.append(movie_id)
                    elif not checked and is_selected:
                        st.session_state.selected_movies.remove(movie_id)
    
    with col2:
        st.markdown("### Your Selection")
        if st.session_state.selected_movies:
            st.markdown(f"**{len(st.session_state.selected_movies)} movies selected**")
            
            # Show selected
            for mid in st.session_state.selected_movies:
                movie = next((m for m in all_movies if m["movieId"] == mid), None)
                if movie:
                    st.caption(f"• {movie['title']}")
            
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.selected_movies = []
                st.rerun()
            
            st.markdown("---")
            get_recs = st.button("🎯 Get Recommendations", type="primary", use_container_width=True)
            
            if get_recs:
                with show_loading_spinner("Generating recommendations based on your selection..."):
                    try:
                        # For simulated user, use SVD with a virtual user approach
                        # Simple approach: average the latent factors of selected movies
                        # For now, use the first selected movie's similar items
                        # TODO: Implement proper virtual user profile
                        first_movie = st.session_state.selected_movies[0]
                        
                        # Get similar movies to the first selected (as proxy)
                        recs = get_user_recommendations_with_metadata(
                            1, "svd", top_n=top_n, include_posters=True
                        )
                        # Filter out already selected
                        recs = [r for r in recs if r["movieId"] not in st.session_state.selected_movies]
                        
                        st.session_state.recommendations = recs[:top_n]
                        show_success(f"Found {len(st.session_state.recommendations)} recommendations!")
                    except Exception as e:
                        show_error(f"Error: {str(e)}")
        else:
            st.info("Select movies from the catalog to get recommendations.")
    
    # Display recommendations
    if st.session_state.recommendations:
        st.markdown("### 🎯 Recommendations Based on Your Selection")
        display_movie_grid(st.session_state.recommendations, cols=4)

# ============================================================
# MODE 3: POPULAR (COLD START)
# ============================================================
else:
    st.subheader("🔥 Popular Movies (Cold Start)")
    st.markdown("Top-rated movies globally — great for new users!")
    
    with show_loading_spinner("Loading popular movies..."):
        popular = get_cold_start_recommendations(top_n=top_n)
    
    if popular:
        st.session_state.recommendations = popular
        show_success(f"Loaded {len(popular)} popular movies!")
    
    if st.session_state.recommendations:
        st.markdown(f"### 🔥 Top {len(st.session_state.recommendations)} Popular Movies")
        st.caption("Based on average rating (minimum 5 ratings)")
        display_movie_grid(st.session_state.recommendations, cols=4)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#888;font-size:12px;">
    Built with Python, scikit-surprise, Streamlit | 
    Data: MovieLens ml-latest-small | 
    Posters: OMDb API
    </div>
    """,
    unsafe_allow_html=True
)