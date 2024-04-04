import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

# Sample music data
music_data = {
    'user_id': [1, 1, 1, 2, 2],
    'song_id': [1, 2, 3, 2, 3],
    'rating': [5, 4, 3, 4, 2],
}

# Create a DataFrame
df = pd.DataFrame(music_data)

# Create a user-item matrix
user_item_matrix = df.pivot(index='user_id', columns='song_id', values='rating').fillna(0)

# Convert the user-item matrix to a sparse matrix
sparse_user_item = sparse.csr_matrix(user_item_matrix.values)

# Calculate the cosine similarity
similarities = cosine_similarity(sparse_user_item)

# Convert the cosine similarities into a DataFrame
cosine_sim_df = pd.DataFrame(similarities, index=user_item_matrix.index, columns=user_item_matrix.index)

# Function to get top recommendations for a user
def get_recommendations(user_id, n=5):
    similar_users = cosine_sim_df[user_id].sort_values(ascending=False).index[1:]
    recommendations = []
    for user in similar_users:
        songs = user_item_matrix.loc[user][user_item_matrix.loc[user] > 0].index
        for song in songs:
            if song not in user_item_matrix.loc[user_id] and song not in recommendations:
                recommendations.append(song)
                if len(recommendations) == n:
                    break
        if len(recommendations) == n:
            break
    return recommendations

# Get recommendations for user 1
user_id = 1
recommendations = get_recommendations(user_id)
print("Recommendations for User", user_id, ":")
for i, song_id in enumerate(recommendations):
    print(f"{i+1}. Song {song_id}")
