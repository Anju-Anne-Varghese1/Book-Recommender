from flask import Flask, render_template, request
import pandas as pd
import numpy as np

popular_df = pd.read_pickle('/Users/Lenovo/projects/website hosting/book recommender system9/book-recommender-system9/popular.pkl')
pt = pd.read_pickle('/Users/Lenovo/projects/website hosting/book recommender system9/book-recommender-system9/pt.pkl')
books = pd.read_pickle('/Users/Lenovo/projects/website hosting/book recommender system9/book-recommender-system9/books.pkl')
similarity_scores = pd.read_pickle('/Users/Lenovo/projects/website hosting/book recommender system9/book-recommender-system9/similarity_scores.pkl')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html',
                           book_name=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_ratings'].values),
                           )


@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')


@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')
    ind = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[ind])), key=lambda x: x[1], reverse=True)[1:5]

    data = []

    for i in similar_items:
        item = []

        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)

    print(data)
    return render_template('recommend.html',  data=data)


if __name__ == '__main__':
    app.run(port=9001, debug=True)
