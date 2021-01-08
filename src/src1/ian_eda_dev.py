import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from surprise import SVD, Dataset, NMF
from surprise.model_selection import train_test_split
from surprise.model_selection.search import GridSearchCV
from surprise.model_selection.validation import cross_validate

def custom_rmse_cv():
    pass

if __name__ == '__main__':
    df = pd.read_csv('data/ml-latest-small/ratings.csv')
    
    # ratings = pd.pivot_table(data=df, values='rating', index='userId', columns='movieId')
    ratings = Dataset.load_builtin('ml-100k')
    # train, test = train_test_split(ratings)

    # algo = SVD(n_factors=50)
    # SVD.fit(train)

    # param_grid = {'n_epochs': [20], 'lr_all': [.01], 'n_factors': [183],
    #             'reg_all': [.1], 'verbose': [True]}
    # gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
    # gs.fit(ratings)
    # print(gs.best_params)

    # for e in range(20,31):
    #     print(e)
    #     algo = SVD(n_factors=183, reg_all=.1, lr_all=.01, n_epochs=e)
    #     results = cross_validate(algo, ratings, measures=['RMSE'])
    #     print(np.mean(results['test_rmse']))