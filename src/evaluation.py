"""Evaluation metrics for collaborative filtering."""
import numpy as np
import pandas as pd
from surprise import accuracy
from collections import defaultdict


def precision_recall_at_k(predictions, k=10, threshold=4.0):
    """
    Compute Precision@K and Recall@K.
    
    Args:
        predictions: List of surprise Prediction objects
        k: Number of top recommendations
        threshold: Rating threshold for "relevant" items
    
    Returns:
        Tuple of (precision, recall) averaged over users
    """
    # Map user -> list of (item, true_rating, est_rating)
    user_preds = defaultdict(list)
    for pred in predictions:
        user_preds[pred.uid].append((pred.iid, pred.r_ui, pred.est))
    
    precisions = []
    recalls = []
    
    for uid, preds in user_preds.items():
        # Sort by estimated rating descending
        preds.sort(key=lambda x: x[2], reverse=True)
        
        # Top-k items
        top_k = preds[:k]
        
        # Relevant items (true rating >= threshold)
        relevant = set(iid for iid, true_r, _ in preds if true_r >= threshold)
        recommended = set(iid for iid, _, _ in top_k)
        
        # Precision@K
        n_rel_and_rec = len(relevant & recommended)
        precision = n_rel_and_rec / k if k > 0 else 0
        
        # Recall@K
        n_rel = len(relevant)
        recall = n_rel_and_rec / n_rel if n_rel > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    return np.mean(precisions), np.mean(recalls)


def coverage(predictions, trainset, k=10):
    """
    Compute catalog coverage: % of items that appear in top-k recommendations.
    """
    # Get all items recommended in top-k for any user
    recommended_items = set()
    
    user_preds = defaultdict(list)
    for pred in predictions:
        user_preds[pred.uid].append((pred.iid, pred.est))
    
    for uid, preds in user_preds.items():
        preds.sort(key=lambda x: x[1], reverse=True)
        top_k = preds[:k]
        recommended_items.update(iid for iid, _ in top_k)
    
    n_items = trainset.n_items
    return len(recommended_items) / n_items


def novelty(predictions, trainset, k=10):
    """
    Compute average novelty (inverse popularity) of recommended items.
    Lower popularity = higher novelty.
    """
    # Compute item popularity (number of ratings in trainset)
    item_popularity = defaultdict(int)
    for uid in range(trainset.n_users):
        for iid, _ in trainset.ur[uid]:
            item_popularity[iid] += 1
    
    # Normalize popularity to [0, 1]
    max_pop = max(item_popularity.values()) if item_popularity else 1
    norm_pop = {iid: pop / max_pop for iid, pop in item_popularity.items()}
    
    user_preds = defaultdict(list)
    for pred in predictions:
        user_preds[pred.uid].append((pred.iid, pred.est))
    
    novelties = []
    for uid, preds in user_preds.items():
        preds.sort(key=lambda x: x[1], reverse=True)
        top_k = preds[:k]
        for iid, _ in top_k:
            # Novelty = 1 - normalized popularity
            novelties.append(1 - norm_pop.get(iid, 0))
    
    return np.mean(novelties) if novelties else 0


def evaluate_recommender(algo, testset, trainset, k=10, threshold=4.0):
    """
    Comprehensive evaluation of a recommender.
    
    Returns dict with all metrics.
    """
    # Get predictions
    predictions = algo.test(testset)
    
    # Rating prediction metrics
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    
    # Top-N metrics
    precision, recall = precision_recall_at_k(predictions, k, threshold)
    cov = coverage(predictions, trainset, k)
    nov = novelty(predictions, trainset, k)
    
    return {
        "rmse": rmse,
        "mae": mae,
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        f"coverage@{k}": cov,
        f"novelty@{k}": nov,
    }


def compare_models(models_dict, testset, trainset, k=10, threshold=4.0):
    """
    Compare multiple models.
    
    Args:
        models_dict: {name: algo} dict
        testset: Surprise testset
        trainset: Surprise trainset
        k: Top-K for recommendation metrics
        threshold: Relevance threshold
    
    Returns:
        DataFrame with metrics for each model
    """
    results = []
    for name, algo in models_dict.items():
        print(f"Evaluating {name}...")
        metrics = evaluate_recommender(algo, testset, trainset, k, threshold)
        metrics["model"] = name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    return df.set_index("model")


def popularity_baseline_recommendations(trainset, testset, k=10):
    """
    Generate popularity-based recommendations as baseline.
    
    Args:
        trainset: Surprise trainset
        testset: List of (uid, iid, true_r) tuples
        k: Number of recommendations per user
    
    Returns:
        List of surprise Prediction objects
    """
    # Compute average rating per item in trainset
    item_ratings = defaultdict(list)
    for uid in range(trainset.n_users):
        for iid, rating in trainset.ur[uid]:
            item_ratings[iid].append(rating)
    
    item_avg = {}
    for iid, ratings in item_ratings.items():
        if len(ratings) >= 5:  # Minimum support
            item_avg[iid] = np.mean(ratings)
    
    # Sort items by average rating
    sorted_items = sorted(item_avg.items(), key=lambda x: x[1], reverse=True)
    top_items = [iid for iid, _ in sorted_items[:k * 2]]  # Get more to account for filtering
    
    # Get test users and their true ratings
    test_user_items = defaultdict(list)
    for uid, iid, true_r in testset:
        test_user_items[uid].append((iid, true_r))
    
    # Get train items per user (to exclude)
    train_user_items = defaultdict(set)
    for uid in range(trainset.n_users):
        train_user_items[uid] = set(iid for iid, _ in trainset.ur[uid])
    
    from surprise import Prediction
    
    predictions = []
    for uid, items in test_user_items.items():
        # Get candidate popular items not rated in train
        candidates = [iid for iid in top_items if iid not in train_user_items[uid]]
        
        for iid in candidates[:k]:
            est = item_avg[iid]
            # Find true rating if in test
            true_r = next((r for i, r in items if i == iid), None)
            if true_r is not None:
                pred = Prediction(uid, iid, true_r, est, None)
                predictions.append(pred)
    
    return predictions


if __name__ == "__main__":
    from src.recommender import load_models_and_data
    
    data, svd_algo, knn_algo = load_models_and_data()
    trainset = data["train_data"].build_full_trainset()
    testset = data["test_data"].build_full_trainset().build_testset()
    
    models = {"SVD": svd_algo, "k-NN": knn_algo}
    results = compare_models(models, testset, trainset, k=10, threshold=4.0)
    print("\nModel Comparison:")
    print(results)
    
    # Popularity baseline
    print("\nEvaluating popularity baseline...")
    pop_predictions = popularity_baseline_recommendations(trainset, testset, k=10)
    pop_metrics = evaluate_recommender(
        type("PopBaseline", (), {"test": lambda self, ts: pop_predictions})(),
        testset, trainset, k=10, threshold=4.0
    )
    print(f"Popularity baseline: {pop_metrics}")