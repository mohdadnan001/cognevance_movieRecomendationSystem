import sys
sys.path.append('..')

from src.evaluation import compare_models, popularity_baseline_recommendations, evaluate_recommender
from src.recommender import load_models_and_data

data, svd_algo, knn_algo = load_models_and_data()
trainset = data["train_data"].build_full_trainset()
testset = data["test_data"].build_full_trainset().build_testset()

models = {"SVD": svd_algo, "k-NN": knn_algo}
results = compare_models(models, testset, trainset, k=10, threshold=4.0)
print("Model Comparison:")
print(results.to_string())
print()

pop_predictions = popularity_baseline_recommendations(trainset, testset, k=10)

class PopBaseline:
    def test(self, ts):
        return pop_predictions

pop_metrics = evaluate_recommender(PopBaseline(), testset, trainset, k=10, threshold=4.0)
print("Popularity baseline:", pop_metrics)