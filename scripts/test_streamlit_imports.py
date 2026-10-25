import sys
sys.path.append('.')

# Test imports only (no data loading)
print("Testing imports...")

from src.recommendation_engine import (
    get_user_recommendations_with_metadata,
    get_cold_start_recommendations,
    get_user_rating_history,
)
print("Recommendation engine imports OK")

from app.utils import (
    display_movie_grid,
    load_user_options,
    get_all_movies_for_simulated_user,
)
print("App utils imports OK")

# Quick sanity check
print("Testing function signatures...")
import inspect
sig = inspect.signature(get_user_recommendations_with_metadata)
print(f"  get_user_recommendations_with_metadata: {sig}")

sig = inspect.signature(get_cold_start_recommendations)
print(f"  get_cold_start_recommendations: {sig}")

print("All import tests passed!")