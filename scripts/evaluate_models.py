"""Evaluate the SVD recommender and save metrics for the app.

Run from anywhere:  python scripts/evaluate_models.py
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation import PopularityBaseline, evaluate_recommender, save_metrics
from src.recommender import load_models_and_data


def main() -> int:
    data, svd_algo = load_models_and_data()
    trainset = data["train_data"].build_full_trainset()
    testset = data["test_data"].build_full_trainset().build_testset()

    print("SVD (full-catalog ranking)")
    metrics = evaluate_recommender(svd_algo, testset, trainset, k=10)

    print("\nPopularity baseline (reference only)")
    evaluate_recommender(PopularityBaseline(trainset), testset, trainset, k=10)

    print(f"\nSaved metrics to {save_metrics(metrics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
