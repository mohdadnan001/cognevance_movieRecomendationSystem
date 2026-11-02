"""Evaluation metrics for the SVD recommender.

Ranking metrics (Precision@K, Recall@K, Coverage@K, Novelty@K) are computed over
the **full catalog**: for every user we score all items they did not rate in the
training set and keep the top-K. Ranking only the held-out test items - which is
what this module used to do - inflates precision by roughly two orders of
magnitude and cannot detect the classic matrix-factorisation failure where every
user receives the same popular titles.

Rating-prediction metrics (RMSE, MAE) use the held-out test ratings directly.
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from surprise import accuracy

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
METRICS_PATH = PROCESSED_DIR / "metrics.json"


def _user_scores(algo, trainset, uid_inner: int) -> np.ndarray:
    """Estimated rating for every item in the trainset for one user.

    For SVD this uses the factor matrices directly, which is exact (it matches
    ``algo.predict`` on raw IDs) and orders of magnitude faster than calling
    ``predict`` once per item.
    """
    if all(hasattr(algo, attr) for attr in ("pu", "qi", "bu", "bi")):
        scores = (
            trainset.global_mean
            + algo.bu[uid_inner]
            + algo.bi
            + algo.qi @ algo.pu[uid_inner]
        )
        lo, hi = trainset.rating_scale
        return np.clip(scores, lo, hi)

    if hasattr(algo, "fixed_scores"):
        return np.asarray(algo.fixed_scores, dtype=float)

    raw_uid = trainset.to_raw_uid(uid_inner)
    return np.array(
        [
            algo.predict(raw_uid, trainset.to_raw_iid(i), verbose=False).est
            for i in range(trainset.n_items)
        ]
    )


def rank_candidates(algo, trainset, user_id: int, k: int = 10, exclude_rated: bool = True):
    """Top-K full-catalog recommendations for one user.

    Args:
        algo: Trained surprise algorithm (or popularity baseline).
        trainset: Surprise trainset the algorithm was fitted on.
        user_id: Raw user ID.
        k: Number of recommendations.
        exclude_rated: Drop items the user already rated in the trainset.

    Returns:
        ``(item_ids, scores)``: raw MovieLens movie IDs and their predicted
        ratings, sorted best-first. Both empty if the user is unknown.
    """
    try:
        uid_inner = trainset.to_inner_uid(user_id)
    except ValueError:
        return [], []

    scores = _user_scores(algo, trainset, uid_inner).astype(float)
    if exclude_rated:
        for iid_inner, _ in trainset.ur[uid_inner]:
            scores[iid_inner] = -np.inf

    k = min(k, int(np.isfinite(scores).sum()))
    if k <= 0:
        return [], []

    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]

    return [trainset.to_raw_iid(i) for i in top], [float(scores[i]) for i in top]


def _relevant_items(testset, threshold: float) -> dict:
    """Map raw user ID -> set of raw item IDs rated >= threshold in the testset."""
    relevant = defaultdict(set)
    for uid, iid, true_r in testset:
        if true_r >= threshold:
            relevant[uid].add(iid)
    return relevant


def precision_recall_at_k(algo, trainset, testset, k: int = 10, threshold: float = 4.0):
    """Precision@K and Recall@K over the full catalog.

    A recommendation is "relevant" when the user actually rated that movie
    >= ``threshold`` in the held-out test set. Users with no relevant test items
    are skipped (recall is undefined for them).

    Returns:
        Tuple ``(precision, recall, n_users)``.
    """
    relevant = _relevant_items(testset, threshold)

    precisions, recalls = [], []
    for uid, rel_items in relevant.items():
        recommended, _ = rank_candidates(algo, trainset, uid, k=k)
        if not recommended:
            continue
        hits = len(rel_items.intersection(recommended))
        precisions.append(hits / k)
        recalls.append(hits / len(rel_items))

    if not precisions:
        return 0.0, 0.0, 0

    return float(np.mean(precisions)), float(np.mean(recalls)), len(precisions)


def coverage(algo, trainset, users, k: int = 10) -> float:
    """Catalog coverage: share of the catalog that appears in some top-K list."""
    recommended = set()
    for uid in users:
        items, _ = rank_candidates(algo, trainset, uid, k=k)
        recommended.update(items)

    return len(recommended) / trainset.n_items if trainset.n_items else 0.0


def novelty(algo, trainset, users, k: int = 10) -> float:
    """Mean novelty (1 - normalised popularity) of the recommended items."""
    popularity = np.zeros(trainset.n_items)
    for uid_inner in range(trainset.n_users):
        for iid_inner, _ in trainset.ur[uid_inner]:
            popularity[iid_inner] += 1

    max_pop = popularity.max() if popularity.size else 0
    if max_pop == 0:
        return 0.0
    norm_pop = popularity / max_pop

    novels = []
    for uid in users:
        try:
            uid_inner = trainset.to_inner_uid(uid)
        except ValueError:
            continue
        items, _ = rank_candidates(algo, trainset, uid, k=k)
        for raw_iid in items:
            novels.append(1.0 - norm_pop[trainset.to_inner_iid(raw_iid)])

    return float(np.mean(novels)) if novels else 0.0


def evaluate_recommender(algo, testset, trainset, k: int = 10, threshold: float = 4.0, verbose: bool = True):
    """Full evaluation of the SVD recommender.

    Returns:
        Dict with ``rmse``, ``mae``, ``precision@K``, ``recall@K``,
        ``coverage@K``, ``novelty@K`` and ``n_users``.
    """
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    precision, recall, n_users = precision_recall_at_k(
        algo, trainset, testset, k=k, threshold=threshold
    )
    users = sorted(_relevant_items(testset, threshold))

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        f"coverage@{k}": coverage(algo, trainset, users, k=k),
        f"novelty@{k}": novelty(algo, trainset, users, k=k),
        "n_users": n_users,
        "k": k,
        "threshold": threshold,
    }

    if verbose:
        print(f"RMSE {rmse:.4f} | MAE {mae:.4f}")
        print(
            f"Precision@{k} {precision:.4f} | Recall@{k} {recall:.4f} | "
            f"Coverage@{k} {metrics[f'coverage@{k}']:.3f} | "
            f"Novelty@{k} {metrics[f'novelty@{k}']:.3f} ({n_users} users)"
        )

    return metrics


class PopularityBaseline:
    """Rank items by their mean train rating (min ``min_support`` ratings).

    Used as a reference point. It exposes ``test()`` so it can be evaluated with
    the same functions as a trained SVD model.
    """

    def __init__(self, trainset, min_support: int = 5):
        sums = np.zeros(trainset.n_items)
        counts = np.zeros(trainset.n_items)
        for uid_inner in range(trainset.n_users):
            for iid_inner, rating in trainset.ur[uid_inner]:
                sums[iid_inner] += rating
                counts[iid_inner] += 1

        scores = np.full(trainset.n_items, trainset.rating_scale[0], dtype=float)
        supported = counts >= min_support
        scores[supported] = sums[supported] / counts[supported]

        self.fixed_scores = scores
        self.trainset = trainset

    def predict(self, uid, iid, verbose=False):
        return self.fixed_scores[self.trainset.to_inner_iid(iid)]

    def test(self, testset):
        from surprise import Prediction

        return [
            Prediction(uid, iid, true_r, self.predict(uid, iid), None)
            for uid, iid, true_r in testset
        ]


def save_metrics(metrics: dict, path: Path = METRICS_PATH) -> Path:
    """Persist evaluation metrics so the app can display them without recomputing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def load_metrics(path: Path = METRICS_PATH):
    """Load saved metrics, or ``None`` if evaluation has not been run yet."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


if __name__ == "__main__":
    from src.recommender import load_models_and_data

    data, svd_algo, _ = load_models_and_data()
    trainset = data["train_data"].build_full_trainset()
    testset = data["test_data"].build_full_trainset().build_testset()

    print("SVD (full-catalog ranking)")
    svd_metrics = evaluate_recommender(svd_algo, testset, trainset, k=10)

    print("\nPopularity baseline (reference)")
    evaluate_recommender(PopularityBaseline(trainset), testset, trainset, k=10)

    print(f"\nSaved metrics to {save_metrics(svd_metrics)}")
