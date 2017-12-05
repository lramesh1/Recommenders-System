# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    pass
    movies['tokens'] = np.nan
    tokens = [tokenize_string(value) for value in movies.genres]
    movies['tokens'] = tokens
    return movies


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i
    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO
    pass
    movies['features'] = np.nan
    pd.set_option('chained_assignment', None)
    tokens_list = movies['tokens'].tolist()
    result = []
    [result.extend(el) for el in tokens_list]
    all_result = Counter(result)
    result = set(result)
    vocab = defaultdict(lambda: len(vocab))
    for i in sorted(result):
        vocab[i]
    a = sorted(result)
    indices = (1, len(a))
    counts = np.zeros(indices)
    number = np.zeros(indices)
    for k, v in all_result.items():
        for i in a:
            if i == k:
                z = a.index(k)
                counts[0, z] = v
    counts = csr_matrix(counts)
    for i in range(0, len(a)):
        number[0, i] = len(movies)

    number = csr_matrix(number)
    res = number / counts
    res.data = np.log10(res.data)
    res = csr_matrix(res)

    for i in range(0, len(movies)):
        matrix = np.zeros(indices)
        maximum = np.zeros(indices)
        for k in movies['tokens'][i]:
            if a.__contains__(k):
                z = a.index(k)
                matrix[0, z] += 1
            inter = csr_matrix(matrix)
            maxi = max(inter.data)
            for s in range(0, len(a)):
                maximum[0, s] = maxi
            maximum = csr_matrix(maximum)
            inter1 = inter / maximum
            inter1 = csr_matrix(inter1)
            inter1 = inter1.multiply(res)
            inter1 = inter1.toarray()
            inter1 = np.nan_to_num(inter1)
            movies['features'][i] = csr_matrix(inter1)
    return movies, vocab

def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    pass
    v1 = np.sqrt(a * a.transpose())
    v2 = np.sqrt(b * b.transpose())
    cos_sim = a.dot(b.transpose()) / (v1 * v2)
    a = cos_sim[0, 0]
    return a


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.
    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.
    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.
    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    pass
    new_list = []
    for i in ratings_test.itertuples():
        sim = []
        cos1 = 0
        rat = []
        for j in ratings_train.itertuples():
            if i[1] == j[1]:
                d = np.where(movies.movieId == i[2])
                b = (movies.features.iloc[d])
                for u, v in b.iteritems():
                    b = v
                ratings = j[3]
                e = np.where(movies.movieId == j[2])
                a = (movies.features.iloc[e])
                for u, v in a.iteritems():
                    a = v
                cos = cosine_sim(a, b)

                if cos > 0.0:
                    sim.append(cos)
                    cos1 += (cos * ratings)
                else:
                    rat.append(ratings)
        if not sim:
            rat = np.array(rat)
            wt_avg = np.mean(rat)
        else:
            sim = np.array(sim)
            sum = np.sum(sim)
            wt_avg = cos1 / sum
        #print(wt_avg)
        new_list.append(wt_avg)
    return np.array(new_list)



def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
