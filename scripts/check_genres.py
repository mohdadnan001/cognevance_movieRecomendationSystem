import sys
sys.path.append('..')

from src.preprocessing import load_preprocessed_data

data = load_preprocessed_data()
movies = data['movies_clean']

print('Sample genres_list:')
for i in range(3):
    val = movies.iloc[i]['genres_list']
    print(f"  movieId={movies.iloc[i]['movieId']}, genres_list={val}, type={type(val)}")

# Check if it's stored as string
print("\nFirst 5 genres_list values:")
print(movies['genres_list'].head())