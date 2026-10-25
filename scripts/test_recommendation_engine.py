import sys
sys.path.append('..')

from src.recommendation_engine import (
    get_user_recommendations_with_metadata,
    get_cold_start_recommendations,
    get_user_rating_history,
)

print("Testing recommendations for user 1 (SVD)...")
recs = get_user_recommendations_with_metadata(1, "svd", top_n=5)
for r in recs:
    print(f"  {r['title']} ({r['year']}) - Est: {r['estimated_rating']} - Genres: {r['genres']}")

print("\nTesting cold start recommendations...")
cold = get_cold_start_recommendations(5)
for r in cold:
    print(f"  {r['title']} ({r['year']}) - Avg: {r['average_rating']} - Genres: {r['genres']}")

print("\nTesting user 1 rating history...")
hist = get_user_rating_history(1, top_n=5)
for r in hist:
    print(f"  {r['title']} ({r['year']}) - Rating: {r['rating']} - Genres: {r['genres']}")

print("\nTesting k-NN recommendations...")
recs_knn = get_user_recommendations_with_metadata(1, "knn", top_n=5)
for r in recs_knn:
    print(f"  {r['title']} ({r['year']}) - Est: {r['estimated_rating']} - Genres: {r['genres']}")

print("\nAll tests passed!")