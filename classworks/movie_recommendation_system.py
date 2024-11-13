# -*- coding: utf-8 -*-
"""Movie Recommendation System.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ohACraddP9t3NyHhnIlAmIgZ-_xnrRqS
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam

# Load ratings and movies data
ratings = pd.read_csv("ratings.csv")   # Contains userId, movieId, rating
movies = pd.read_csv("movies.csv")     # Contains movieId, title

# Map movie IDs to titles
movie_id_to_title = dict(zip(movies['movieId'], movies['title']))

# Encode userId and movieId as categorical variables for embeddings
user_ids = ratings['userId'].unique()
movie_ids = ratings['movieId'].unique()

user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
movie_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

ratings['user'] = ratings['userId'].map(user_to_index)
ratings['movie'] = ratings['movieId'].map(movie_to_index)

num_users = len(user_to_index)
num_movies = len(movie_to_index)

# Prepare features and target
X = ratings[['user', 'movie']].values
y = ratings['rating'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define input layers for users and movies
user_input = Input(shape=(1,), name='user_input')
movie_input = Input(shape=(1,), name='movie_input')

# Embedding layers for users and movies
user_embedding = Embedding(input_dim=num_users, output_dim=50, name='user_embedding')(user_input)
movie_embedding = Embedding(input_dim=num_movies, output_dim=50, name='movie_embedding')(movie_input)

# Flatten embeddings
user_vec = Flatten(name='user_flatten')(user_embedding)
movie_vec = Flatten(name='movie_flatten')(movie_embedding)

# Concatenate user and movie embeddings
concat = Concatenate()([user_vec, movie_vec])

# Fully connected MLP layers
fc1 = Dense(128, activation='relu')(concat)
fc2 = Dense(64, activation='relu')(fc1)
fc3 = Dense(32, activation='relu')(fc2)
output = Dense(1)(fc3)  # Regression output for rating prediction

# Build the model
model = Model(inputs=[user_input, movie_input], outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
# Train the model
history = model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
# Function to recommend movies with names for a given user
def recommend_movies(user_id, num_recommendations=5):
    if user_id not in user_to_index:
        print(f"User ID {user_id} not found in the data.")
        return []

    user_idx = user_to_index[user_id]
    # Create a list of all movie indices for the user
    user_movie_pairs = np.array([[user_idx, movie_idx] for movie_idx in range(num_movies)])

    # Predict ratings for all movies for this user
    predicted_ratings = model.predict([user_movie_pairs[:, 0], user_movie_pairs[:, 1]]).flatten()

    # Get top movie indices based on predicted ratings
    top_movie_indices = predicted_ratings.argsort()[-num_recommendations:][::-1]
    recommended_movie_ids = [movie_ids[i] for i in top_movie_indices]
    recommended_movie_names = [movie_id_to_title[movie_id] for movie_id in recommended_movie_ids]

    return recommended_movie_names

# Get recommendations based on real-time user input
user_id = int(input("Enter User ID to get recommendations: "))
recommended_movies = recommend_movies(user_id)
print(f"Recommended movies for user {user_id}: {recommended_movies}")