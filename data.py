import numpy as np
import pandas as pd
book_df = pd.read_csv('Data/Books.csv')
ratings_df=pd.read_csv('Data/Ratings.csv')
user_df=pd.read_csv('Data/Users.csv')
user_rating_df = ratings_df.merge(user_df, left_on = 'User-ID', right_on = 'User-ID')
book_user_rating = book_df.merge(user_rating_df, left_on = 'ISBN',right_on = 'ISBN')
book_user_rating = book_user_rating[['ISBN', 'Book-Title', 'Book-Author', 'User-ID', 'Book-Rating']]
book_user_rating.reset_index(drop=True, inplace = True)
d ={}
for i,j in enumerate(book_user_rating.ISBN.unique()):
    d[j] =i
book_user_rating['unique_id_book'] = book_user_rating['ISBN'].map(d)
users_books_pivot_matrix_df = book_user_rating.pivot(index='User-ID', columns='unique_id_book', values='Book-Rating').fillna(0)
users_books_pivot_matrix_df.to_csv('Data/matrix_pd.csv')