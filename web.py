import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model from the pkl file
with open('deep_model_data.pkl', 'rb') as f:
    similar_user_movies,ranked_item_score,similar_users,ranked_item_score = pickle.load(f)



def predict_similar_movies(userId,  movie_count):
    item_score = {}
    for i in similar_user_movies.columns:
        movie_rating = similar_user_movies[i]
        total = 0
        count = 0
        for u in similar_users.index:
            if pd.isna(movie_rating[u]) == False:
                score = similar_users[u] * movie_rating[u]
                total += score
                count +=1
                item_score[i] = total / count


    item_score = pd.DataFrame(item_score.items(), columns=['movie', 'movie_score'])
    ranked_item_score = item_score.sort_values(by='movie_score', ascending=False)



    
    st.write(f"Here are {movie_count} movie recommendations for user {userId} :")
    st.table(ranked_item_score.head(movie_count))

# Create the Streamlit app
st.title(' Movie Recommender')

# Get user input
userId = st.number_input('Enter your user ID', min_value=1, max_value=610, value=1)
movie_count = st.number_input('Enter the number of recommendations you want', min_value=1, max_value=25, value=10)

# Make recommendations based on user input
if st.button('Get Recommendations'):
    predict_similar_movies(userId,movie_count)