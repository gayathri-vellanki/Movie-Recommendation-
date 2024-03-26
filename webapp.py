import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Movie Recommendation System")
st.markdown('Using Collabrative Filtering')
st.write("For Movies Dataset: Check out this [link](https://github.com/gayathri-vellanki/CoApps-/blob/main/movies.csv)")
DATA_URL=(r'movies.csv')
@st.cache_data(persist=True)
def load_data(nrows):
    data=pd.read_csv(r'movies.csv',nrows=nrows)
    lowercase=lambda x: str(x).lower()
    data.rename(lowercase,axis='columns',inplace=True)
    return data
data=load_data(10000000)
original_data=data
if st.checkbox('Show Movies Data'):
    st.subheader('MOVIES RAW DATA')
    st.write(data)
st.write("For Ratings Dataset: Check out this [link](https://github.com/gayathri-vellanki/CoApps-/blob/main/ratings.csv)")
DATA_URL=(r'ratings.csv')
@st.cache_data(persist=True)
def load_data(nrows):
    data=pd.read_csv(r'ratings.csv',nrows=nrows)
    lowercase=lambda x: str(x).lower()
    data.rename(lowercase,axis='columns',inplace=True)
    return data
data=load_data(10000)
original_data=data
if st.checkbox('Show Ratings Data'):
    st.subheader('MOVIE RATINGS DATA')
    st.write(data)
    
import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns
# Similarity
from sklearn.metrics.pairwise import cosine_similarity
# Read in data
ratings=pd.read_csv(r'ratings.csv')
# Read in data
movies = pd.read_csv(r'movies.csv')

# Take a look at the data
movies.head()

# Merge ratings and movies datasets
df = pd.merge(ratings, movies, on='movieId', how='inner')
st.subheader("Merge Movie and Ratings data")
st.write(df)
# Take a look at the data
df.head()
agg_ratings = df.groupby('title').agg(mean_rating = ('rating', 'mean'),number_of_ratings = ('rating', 'count')).reset_index()
agg_ratings_GT100 = agg_ratings[agg_ratings['number_of_ratings']>100]

# Check popular movies
popular=agg_ratings_GT100.sort_values(by='number_of_ratings', ascending=False).head()
st.write("POPULAR MOVIES")
st.write(popular)
st.header('--VISULIZATION--')
st.write("jointplot to check the correlation between the average rating and the number of ratings.We can see an upward trend from the scatter plot, showing that popular movies get higher ratings.The average rating distribution shows that most movies in the dataset have an average rating of around 4.The number of rating distribution shows that most movies have less than 150 ratings.")
# Visulization
fig=sns.jointplot(x='mean_rating', y='number_of_ratings', data=agg_ratings_GT100)
st.pyplot(fig)
df_GT100 = pd.merge(df, agg_ratings_GT100[['title']], on='title', how='inner')
rate=sorted(df_GT100['rating'].unique())
if st.checkbox('The Unique ratings are:'):
    for i in rate:
        st.write(i,end=",") 
matrix = df_GT100.pivot_table(index='userId', columns='title', values='rating')
matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 'rows')
user_similarity = matrix_norm.T.corr()
user_similarity_cosine = cosine_similarity(matrix_norm.fillna(0))
n = 10
user_similarity_threshold = 0.3
st.write("picked userId:")
picked_userid=
st.write(picked_userid)

similar_users = user_similarity[user_similarity[picked_userid]>user_similarity_threshold][picked_userid].sort_values(ascending=False)[:n]
picked_userid_watched = matrix_norm[matrix_norm.index == picked_userid].dropna(axis=1, how='all')
similar_user_movies = matrix_norm[matrix_norm.index.isin(similar_users.index)].dropna(axis=1, how='all')
similar_user_movies.drop(picked_userid_watched.columns,axis=1, inplace=True, errors='ignore')
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
m = 10
ranked_item_score.head(m)
avg_rating = matrix[matrix.index == picked_userid].T.mean()[picked_userid]

# Calcuate the predicted rating
ranked_item_score['predicted_rating'] = ranked_item_score['movie_score'] + avg_rating
st.header('Recommend the top 10 movies to user')
ranked_item_score.head(m)
if st.checkbox('Recommended Movies'):
    st.subheader('Movies Recommended are:')
    st.write(ranked_item_score.head(m))