"""Smoke-test the recommendation engine (SVD, cold start, history, simulated user).

Run from anywhere:  python scripts/test_recommendation_engine.py
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.recommendation_engine import (
    get_cold_start_recommendations,
    get_simulated_user_recommendations,
    get_user_rating_history,
    get_user_recommendations_with_metadata,
)


def main() -> int:
    print("User 1 (SVD):")
    for r in get_user_recommendations_with_metadata(1, "svd", top_n=5):
        print(f"  {r['title']} ({r['year']}) - Est: {r['estimated_rating']} - {r['genres']}")

    print("\nCold start:")
    for r in get_cold_start_recommendations(5):
        print(f"  {r['title']} ({r['year']}) - Avg: {r['average_rating']} - {r['genres']}")

    print("\nUser 1 history:")
    for r in get_user_rating_history(1, top_n=5):
        print(f"  {r['title']} ({r['year']}) - Rating: {r['rating']} - {r['genres']}")

    print("\nSimulated user (likes Toy Story and Jumanji):")
    for r in get_simulated_user_recommendations([1, 2], top_n=5):
        print(f"  {r['title']} ({r['year']}) - Est: {r['estimated_rating']} - {r['genres']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
