# importing libraries
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# importing 3 datasets
books = pd.read_csv('/Users/Lenovo/projects/website hosting/book recommender system9/Books.csv', low_memory=False)
ratings = pd.read_csv('/Users/Lenovo/projects/website hosting/book recommender system9/Ratings.csv')
users = pd.read_csv('/Users/Lenovo/projects/website hosting/book recommender system9/Users.csv')

# we are going to merge ratings and books on 'ISBN' column
ratings_with_name = ratings.merge(books, on='ISBN')

# done reset_index() to obtain the dataframe
num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()

# renaming a particular column
num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)

# done reset_index() to obtain the dataframe
avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()

# renaming a particular column
avg_rating_df.rename(columns={'Book-Rating': 'avg_ratings'}, inplace=True)

popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')

# top 50 books on the platform
popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_ratings', ascending=False).head(50)

popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')

popular_df = popular_df[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_ratings']]

# round values in 'avg_ratings' column to two decimal places
popular_df['avg_ratings'] = popular_df['avg_ratings'].round(2)

# storing it in a new variable
x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200

padhe_likhe_users = x[x].index

filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]

y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50

famous_books = y[y].index

final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')

pt.fillna(0, inplace=True)

similarity_scores = cosine_similarity(pt)

# saving the files in binary format using pickle
pickle.dump(popular_df, open('popular.pkl', 'wb'))
pickle.dump(pt, open('pt.pkl', 'wb'))
pickle.dump(books, open('books.pkl', 'wb'))
pickle.dump(similarity_scores, open('similarity_scores.pkl', 'wb'))
