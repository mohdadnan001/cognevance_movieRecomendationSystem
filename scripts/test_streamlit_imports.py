"""Import smoke test for the Streamlit app modules (no data loading).

Run from anywhere:  python scripts/test_streamlit_imports.py
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import inspect

from src.evaluation import evaluate_recommender, load_metrics
from src.recommendation_engine import (
    get_cold_start_recommendations,
    get_simulated_user_recommendations,
    get_user_rating_history,
    get_user_recommendations_with_metadata,
)


def main() -> int:
    print("Source imports OK")

    for func in (
        get_user_recommendations_with_metadata,
        get_cold_start_recommendations,
        get_user_rating_history,
        get_simulated_user_recommendations,
        evaluate_recommender,
    ):
        print(f"  {func.__name__}{inspect.signature(func)}")

    print(f"  load_metrics() -> {load_metrics() is not None and 'saved metrics found' or 'no saved metrics yet'}")
    print("\nAll import tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
