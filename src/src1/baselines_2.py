#!/usr/bin/env python

"""
http://surprise.readthedocs.io/en/stable/building_custom_algo.html
"""

import sys
import numpy as np
from surprise import AlgoBase, Dataset
from surprise.model_selection import train_test_split
from surprise.model_selection.validation import cross_validate
import pandas as pd

from scipy.sparse import csr_matrix

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('data/ml-latest-small/ratings.csv')
class_as_mat = csr_matrix((df.rating, ((df.userId), (df.movieId))))



class GlobalMean(AlgoBase):
    def __init__(self):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)

    def fit(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.fit(self, trainset)

        # Compute the average rating. We might as well use the
        # trainset.global_mean attribute ;)
        self.the_mean = np.mean([r for (_, _, r) in
                                 self.trainset.all_ratings()])

        return self

    def estimate(self, u, i):

        return self.the_mean


class MeanofMeans(AlgoBase):
    def __init__(self):

    # Always call base method before doing anything.
        AlgoBase.__init__(self)


    def fit(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.fit(self, trainset)

        self.users = np.array([u for (u, _, _) in self.trainset.all_ratings()])
        self.items = np.array([i for (_, i, _) in self.trainset.all_ratings()])
        self.ratings = np.array([r for (_, _, r) in self.trainset.all_ratings()])

        user_means,item_means = {},{}
        for user in np.unique(self.users):
            user_means[user] = self.ratings[self.users==user].mean()
        for item in np.unique(self.items):
            item_means[item] = self.ratings[self.items==item].mean()

        self.global_mean = self.ratings.mean()
        self.user_means = user_means
        self.item_means = item_means

    def estimate(self, u, num):
        """
        return the mean of means estimate
        """
        predictions = []
        for i in np.unique(self.items):
            prediction = np.mean([self.global_mean,
                        self.user_means[u],
                        self.item_means[i]])
            predictions.append(prediction)
        sort = np.argsort(predictions)
        sorted_movies = self.items[sort]
        return sorted_movies[-1:-(num+1):-1]


        # if u not in self.user_means:
        #     return(np.mean([self.global_mean,
        #                     self.item_means[i]]))

        # if i not in self.item_means:
        #     return(np.mean([self.global_mean,
        #                     self.user_means[u]]))

        # return(np.mean([self.global_mean,
        #                 self.user_means[u],
        #                 self.item_means[i]]))


if __name__ == "__main__":

    data = Dataset.load_builtin('ml-100k')

    # print("\nGlobal Mean...")
    # algo = GlobalMean()
    # print(np.mean(cross_validate(algo, data)['test_rmse']))

    # print("\nMeanOfMeans...")
    # algo = MeanofMeans()
    # print(np.mean(cross_validate(algo, data)['test_rmse']))
    
    # print(df.head())

    train, test = train_test_split(data, random_state=20)

    algo = MeanofMeans()
    algo.fit(train)
    recs = algo.estimate(56, 5)
    
    names = pd.read_csv('data/ml-latest-small/movies.csv').set_index('movieId')
    
    for rec in recs:
        print(names.iloc[rec, 0])

    # genres = names['genres'].reset_index().drop('movieId', axis=1)
    # for mov in range(genres.shape[0])