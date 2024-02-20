import pandas as pd

# Load movie information
movies_df = pd.read_csv('Netflix_Dataset_Movie.csv')

# Load user ratings
ratings_df = pd.read_csv('Netflix_Dataset_Rating.csv')

# Merge movie information with user ratings
merged_df = pd.merge(ratings_df, movies_df, on='Movie_ID')

# Creating a user-item matrix
user_item_matrix = merged_df.pivot_table(index='User_ID', columns='Name', values='Rating')

# Function to recommend items to a user based on collaborative filtering
def recommend_items(user_id):
    try:
        # Get the ratings of the user
        user_ratings = user_item_matrix.loc[user_id]
    except KeyError:
        print(f"User ID {user_id} not found.")
        return None
    
    # Calculate the correlation with other users
    similar_users = user_item_matrix.corrwith(user_ratings, axis=0, drop=True).sort_values(ascending=False)
    
    # Get the items that the user hasn't rated yet
    unrated_items = user_ratings[user_ratings.isnull()].index
    
    # Predict ratings for unrated items based on similar users
    recommendations = pd.Series(index=unrated_items)
    for item in unrated_items:
        item_ratings = user_item_matrix[item]
        similar_ratings = item_ratings.dropna().index.intersection(similar_users.index)
        if len(similar_ratings) == 0:
            continue  # Skip items with no similar users who have rated them
        recommendations[item] = sum(similar_users.loc[similar_user] * item_ratings.loc[similar_user] for similar_user in similar_ratings) / len(similar_ratings)
    
    # Sort the recommendations by predicted ratings
    recommendations = recommendations.sort_values(ascending=False)
    
    return recommendations

# Example usage
user = 712664 # Assuming user IDs are numeric
recommended_items = recommend_items(user)
print(f"Recommended items for user {user}:")
print(recommended_items.head())
