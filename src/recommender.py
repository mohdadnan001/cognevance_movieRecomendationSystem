"""Collaborative filtering model training and recommendation generation."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from surprise import SVD, KNNBasic, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split as surprise_train_test_split

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


def train_knn(dataset, param_grid=None, cv=3, n_jobs=-1):
    """Train item-based k-NN with grid search."""
    if param_grid is None:
        param_grid = {
            "k": [20, 40, 60],
            "min_k": [1, 3],
            "sim_options": {
                "name": ["cosine", "msd", "pearson"],
                "user_based": [False],  # Item-based
                "min_support": [1, 3],
            },
        }
    
    print("Running GridSearchCV for k-NN...")
    gs = GridSearchCV(
        KNNBasic,
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
    
    best_algo = gs.best_estimator["rmse"]
    trainset = dataset.build_full_trainset()
    best_algo.fit(trainset)
    
    return best_algo, gs


def evaluate_model(algo, testset, verbose=True):
    """Evaluate model on testset."""
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=verbose)
    mae = accuracy.mae(predictions, verbose=verbose)
    return {"rmse": rmse, "mae": mae, "predictions": predictions}


def get_top_n_recommendations(algo, user_id, trainset, n=10, exclude_rated=True):
    """
    Get top-N recommendations for a user.
    
    Args:
        algo: Trained surprise algorithm
        user_id: Raw user ID
        trainset: Surprise trainset (for inner/outer ID conversion)
        n: Number of recommendations
        exclude_rated: Whether to exclude already-rated items
    
    Returns:
        List of (movie_inner_id, estimated_rating) tuples
    """
    # Convert raw user ID to inner ID
    try:
        uid_inner = trainset.to_inner_uid(user_id)
    except ValueError:
        # User not in trainset (cold start)
        return []
    
    # Get all item inner IDs
    all_item_inner_ids = set(range(trainset.n_items))
    
    if exclude_rated:
        # Get items user has already rated
        rated_items = set([iid for (iid, _) in trainset.ur[uid_inner]])
        candidates = all_item_inner_ids - rated_items
    else:
        candidates = all_item_inner_ids
    
    # Predict ratings for all candidates
    predictions = []
    for iid_inner in candidates:
        est = algo.predict(uid_inner, iid_inner, r_ui=None, clip=True, verbose=False).est
        predictions.append((iid_inner, est))
    
    # Sort by estimated rating descending
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    return predictions[:n]


def get_popularity_recommendations(trainset, n=10):
    """Get top-N globally popular items (for cold start)."""
    # Compute average rating per item
    item_ratings = {}
    for uid_inner in range(trainset.n_users):
        for iid_inner, rating in trainset.ur[uid_inner]:
            if iid_inner not in item_ratings:
                item_ratings[iid_inner] = []
            item_ratings[iid_inner].append(rating)
    
    # Average rating and count
    item_stats = []
    for iid_inner, ratings in item_ratings.items():
        if len(ratings) >= 5:  # Minimum ratings threshold
            avg_rating = np.mean(ratings)
            item_stats.append((iid_inner, avg_rating, len(ratings)))
    
    # Sort by average rating, then by count
    item_stats.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    return [(iid, avg) for iid, avg, _ in item_stats[:n]]


def save_model(algo, name):
    """Save trained model."""
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(algo, path)
    print(f"Model saved to {path}")


def load_model(name):
    """Load trained model."""
    path = MODELS_DIR / f"{name}.pkl"
    return joblib.load(path)


def train_and_save_models(
    train_dataset,
    testset,
    svd_param_grid=None,
    knn_param_grid=None,
    cv=3,
    n_jobs=-1,
):
    """Train and save both SVD and k-NN models."""
    print("=" * 50)
    print("TRAINING SVD")
    print("=" * 50)
    svd_algo, svd_gs = train_svd(train_dataset, svd_param_grid, cv, n_jobs)
    
    print("\n" + "=" * 50)
    print("TRAINING k-NN (item-based)")
    print("=" * 50)
    knn_algo, knn_gs = train_knn(train_dataset, knn_param_grid, cv, n_jobs)
    
    print("\n" + "=" * 50)
    print("EVALUATING ON TEST SET")
    print("=" * 50)
    
    svd_results = evaluate_model(svd_algo, testset)
    knn_results = evaluate_model(knn_algo, testset)
    
    print(f"\nSVD - RMSE: {svd_results['rmse']:.4f}, MAE: {svd_results['mae']:.4f}")
    print(f"k-NN - RMSE: {knn_results['rmse']:.4f}, MAE: {knn_results['mae']:.4f}")
    
    # Save models
    save_model(svd_algo, "algo_svd")
    save_model(knn_algo, "algo_knn")
    
    # Save grid search results for analysis
    joblib.dump(svd_gs, MODELS_DIR / "svd_gridsearch.pkl")
    joblib.dump(knn_gs, MODELS_DIR / "knn_gridsearch.pkl")
    
    return {
        "svd": {"algo": svd_algo, "gs": svd_gs, "test_results": svd_results},
        "knn": {"algo": knn_algo, "gs": knn_gs, "test_results": knn_results},
    }


def load_models_and_data():
    """Load trained models and preprocessed data."""
    data = load_preprocessed_data()
    svd_algo = load_model("algo_svd")
    knn_algo = load_model("algo_knn")
    return data, svd_algo, knn_algo


if __name__ == "__main__":
    # Load preprocessed data
    data = load_preprocessed_data()
    trainset = data["train_data"].build_full_trainset()
    testset = data["test_data"].build_full_trainset().build_testset()
    
    # Train and evaluate
    results = train_and_save_models(trainset, testset)