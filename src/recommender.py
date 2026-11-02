"""SVD collaborative-filtering model: training, prediction and recommendations.

Inner IDs are the indices Surprise assigns to users/items when a trainset is
built; raw IDs are the original MovieLens ``userId`` / ``movieId`` values. All
lookups here go through ``trainset.to_inner_*`` / ``trainset.to_raw_*``. Passing
an inner ID into ``algo.predict`` (or indexing a mapping built elsewhere) silently
returns the wrong movie's score.
"""
from pathlib import Path

import joblib
import numpy as np
from surprise import SVD, accuracy
from surprise.model_selection import GridSearchCV

from src.preprocessing import load_preprocessed_data

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = PROCESSED_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_svd(dataset, param_grid=None, cv=3, n_jobs=-1):
    """Train SVD with grid search for hyperparameter tuning."""
    if param_grid is None:
        param_grid = {
            "n_factors": [50, 100, 150],
            "n_epochs": [20, 30],
            "lr_all": [0.005, 0.01],
            "reg_all": [0.02, 0.1],
        }

    print("Running GridSearchCV for SVD...")
    gs = GridSearchCV(
        SVD,
        param_grid,
        measures=["rmse", "mae"],
        cv=cv,
        n_jobs=n_jobs,
        joblib_verbose=1,
    )
    gs.fit(dataset)

    print(f"Best RMSE: {gs.best_score['rmse']:.4f}")
    print(f"Best MAE: {gs.best_score['mae']:.4f}")
    print(f"Best params: {gs.best_params['rmse']}")

    # Train best model on full dataset
    best_algo = gs.best_estimator["rmse"]
    trainset = dataset.build_full_trainset()
    best_algo.fit(trainset)

    return best_algo, gs


def evaluate_model(algo, testset, verbose=True):
    """Evaluate model on testset (rating-prediction accuracy)."""
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=verbose)
    mae = accuracy.mae(predictions, verbose=verbose)
    return {"rmse": rmse, "mae": mae, "predictions": predictions}


def get_top_n_recommendations(algo, user_id, trainset, n=10, exclude_rated=True):
    """
    Get top-N recommendations for a user.

    Scores every catalog item for the user using the SVD factor matrices, which
    is exact (identical to calling ``algo.predict`` with raw IDs) and fast enough
    to rank the whole catalog on each request.

    Args:
        algo: Trained SVD model
        user_id: Raw MovieLens user ID
        trainset: Surprise trainset the model was fitted on
        n: Number of recommendations
        exclude_rated: Whether to exclude items already rated by the user

    Returns:
        List of (movie_id, estimated_rating) tuples, sorted best-first, where
        ``movie_id`` is the raw MovieLens movie ID.
    """
    try:
        uid_inner = trainset.to_inner_uid(user_id)
    except ValueError:
        # User not in trainset (cold start)
        return []

    scores = (
        trainset.global_mean
        + algo.bu[uid_inner]
        + algo.bi
        + algo.qi @ algo.pu[uid_inner]
    )
    lo, hi = trainset.rating_scale
    scores = np.clip(scores, lo, hi)

    if exclude_rated:
        for iid_inner, _ in trainset.ur[uid_inner]:
            scores[iid_inner] = -np.inf

    n = min(n, int(np.isfinite(scores).sum()))
    if n <= 0:
        return []

    top = np.argpartition(-scores, n - 1)[:n]
    top = top[np.argsort(-scores[top])]

    return [(trainset.to_raw_iid(i), float(scores[i])) for i in top]


def get_popularity_recommendations(trainset, n=10):
    """Get top-N globally popular items, as raw movie IDs (for cold start)."""
    sums = np.zeros(trainset.n_items)
    counts = np.zeros(trainset.n_items)
    for uid_inner in range(trainset.n_users):
        for iid_inner, rating in trainset.ur[uid_inner]:
            sums[iid_inner] += rating
            counts[iid_inner] += 1

    supported = counts >= 5  # Minimum ratings threshold
    scores = np.full(trainset.n_items, -np.inf)
    scores[supported] = sums[supported] / counts[supported]

    n = min(n, int(supported.sum()))
    if n <= 0:
        return []

    top = np.argpartition(-scores, n - 1)[:n]
    top = top[np.argsort(-scores[top])]

    return [(trainset.to_raw_iid(i), float(scores[i])) for i in top]


def save_model(algo, name):
    """Save trained model."""
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(algo, path)
    print(f"Model saved to {path}")


def load_model(name):
    """Load trained model."""
    path = MODELS_DIR / f"{name}.pkl"
    return joblib.load(path)


def train_and_save_models(train_dataset, testset, svd_param_grid=None, cv=3, n_jobs=-1):
    """Train, evaluate and save the SVD model."""
    print("=" * 50)
    print("TRAINING SVD")
    print("=" * 50)
    svd_algo, svd_gs = train_svd(train_dataset, svd_param_grid, cv, n_jobs)

    print("\n" + "=" * 50)
    print("EVALUATING ON TEST SET")
    print("=" * 50)
    svd_results = evaluate_model(svd_algo, testset)
    print(f"SVD - RMSE: {svd_results['rmse']:.4f}, MAE: {svd_results['mae']:.4f}")

    save_model(svd_algo, "algo_svd")
    joblib.dump(svd_gs, MODELS_DIR / "svd_gridsearch.pkl")

    return {"svd": {"algo": svd_algo, "gs": svd_gs, "test_results": svd_results}}


def load_models_and_data():
    """Load preprocessed data and the trained SVD model."""
    data = load_preprocessed_data()
    svd_algo = load_model("algo_svd")
    return data, svd_algo


if __name__ == "__main__":
    data = load_preprocessed_data()
    # GridSearchCV takes the surprise Dataset, not a built trainset.
    train_and_save_models(
        data["train_data"],
        data["test_data"].build_full_trainset().build_testset(),
    )
