import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import surprise
import seaborn as sns
from surprise import AlgoBase, Dataset
from surprise import SVD, Dataset, accuracy 
from surprise.model_selection import train_test_split
from surprise.model_selection.search import RandomizedSearchCV
from surprise.model_selection.validation import cross_validate
# sns.set_theme(style="darkgrid")
plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 200
pd.set_option('display.max_columns', None)

# df = pd.read_csv('../data/ml-latest-small/ratings.csv')
# print(df.head())

if __name__ == "__main__":
    df = pd.read_csv('../data/ml-latest-small/ratings.csv')

    # ratings = Dataset.load_builtin('ml-100k')
    # trainset, testset = train_test_split(ratings, test_size=.25)
    # algo = SVD()

    # # Train the algorithm on the trainset, and predict ratings for the testset
    # algo.fit(trainset)
    # predictions = algo.test(testset)

    # # Then compute RMSE
    # accuracy.rmse(predictions)

    # predictions = algo.fit(trainset).test(testset)
    # # print(rs.best_score)
    # # print(rs.best_params)
    # print(inspect.getsource(surprise.SVD()))

    def var_hist(column, df, dic, color, bin_size):
        """Takes in specifications and produces a graph
        Args:
            colors (str, optional): colors to be assigned to "pallete"
                feature inseaborn. Defaults to 'Set3'.
            bin_size (int, optional): Histogram bin size. Defaults to 32.
        """
        ax = sns.histplot(x=column, data=df, palette=color, bins=bin_size)
        ax.set_xlabel(dic['xlabel'])
        ax.set_ylabel(dic['ylabel'])
        ax.set_title(dic['title'])
        plt.show()


    def snscountplot(column, df, dic, color='aqua'):
        """Creats a histogram-like barchart that sums
            categories in a given column
        Args:
            colors (str, optional): [description]. Defaults to "Set3".
        """
        ax = sns.countplot(x=column, data=df, color=color)
        ax.set_xlabel(dic['xlabel'])
        ax.set_ylabel(dic['ylabel'])
        ax.set_title(dic['title'])
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        plt.show()

    def barplot(x, y, dic, color):
        fig , ax = plt.subplots()
        ax.bar(x, y, color=color)
        ax.set_xlabel(dic['xlabel'])
        ax.set_ylabel(dic['ylabel'])
        ax.set_title(dic['title'])
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        plt.show()

    rat_cp_dic = {
        'title': 'Number of Movies rated per Goblin',
        'xlabel': 'Goblins (1 per line 610 total)',
        'ylabel': '# of Movies Rated'
    }

    # snscountplot('userId', df, rat_cp_dic, 'seagreen')

    movie_hist_dic = {
        'title': 'Ratings Per Movie Histogram',
        'xlabel': 'Movies (1 per Bin)',
        'ylabel': '# of Times Movie Rated'
    }
    snscountplot('movieId', df, movie_hist_dic, 'navajowhite')
    

    df['cat_length'] = 0
    df['rating score'] = 0
    df['total_ratings'] = 0

    excellent_voters = df[df['rating'] == 5.0]
    good_voters = df[(df['rating'] < 5.0) & (df['rating'] > 3.5)]
    avg_voters = df[(df['rating'] < 4.0) & (df['rating'] > 2.0)]
    poor_voters = df[df["rating"] < 2.5]

    voters = [excellent_voters, good_voters, avg_voters, poor_voters]

    # for i in voters:
    #     snscountplot('userId', i, rat_cp_dic, 'seagreen')

    # for cat in voters:
    #     for i in cat.unique():
    #         cat['cat_length'][cat['userId'] == i] = len(cat['movieId'].unique())







    # avg_dic = dict()
    # for user in df['userId'].unique():
    #     avg_dic[user] = df['rating'][df['userId'] == user].mean()
    # avg_dic[620]= 0
    
    # fig, ax = plt.subplots()
    # ax.hist(avg_dic.values(), bins=10, color='seagreen')
    # ax.set_xlabel('Avergae Rating')
    # ax.set_ylabel('# of Goblins')
    # ax.set_title('Average Goblin Rating Histogram')
    # plt.show()

    



    # for i in voters:
    #     print(len(i), len(i['userId'].unique()), "\n")

    # print(len(df['userId'].unique()))
    # print(df.info())
    # df2 = pd.DataFrame()

    # bar = {
    #     'title': 'Movie rating histogram',
    #     'xlabel': 'Goblins (1 per Bin)',
    #     'ylabel': '# of Movies Rated'
    # }
    # df2 = df.groupby('userId').count()
    # df2 = df2.reset_index('userId')
    # df2 = df2.sort_values('rating', ascending=False)
    # print(df2.head())
    
    # barplot(df2['userId'], df2['rating'], bar, 'seagreen')


    # print(df.head())