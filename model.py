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
# Take a look at the data
df.head()
agg_ratings = df.groupby('title').agg(mean_rating = ('rating', 'mean'),number_of_ratings = ('rating', 'count')).reset_index()
agg_ratings_GT100 = agg_ratings[agg_ratings['number_of_ratings']>100]

# Check popular movies
popular=agg_ratings_GT100.sort_values(by='number_of_ratings', ascending=False).head()
# Visulization
fig=sns.jointplot(x='mean_rating', y='number_of_ratings', data=agg_ratings_GT100)
df_GT100 = pd.merge(df, agg_ratings_GT100[['title']], on='title', how='inner')
rate=sorted(df_GT100['rating'].unique())
matrix = df_GT100.pivot_table(index='userId', columns='title', values='rating')
matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 'rows')
user_similarity = matrix_norm.T.corr()
user_similarity_cosine = cosine_similarity(matrix_norm.fillna(0))
n = 10
user_similarity_threshold = 0.3
picked_userid=int(input("Enter userID:"))

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
m = int(input("Enter Number of Recommendatios:"))
print(ranked_item_score.head(m))
