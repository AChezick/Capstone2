import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics 
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
pd.set_option('display.max_columns', None) 
dataframe = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/temp_csv/ab_df.csv') 
ids = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/testing/x1.csv')

def pca():
# Code Adopted from https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html #
    ids2 = ids.drop(['Patient_ID', 'prediction', 'Proba'], axis=1)

    y = ids2['y_target']
    X = ids2.copy() 

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.30,
                                                        random_state=101)

    unscaled_clf = make_pipeline(PCA(n_components=5), GaussianNB())
    unscaled_clf.fit(X_train, y_train)
    pred_test = unscaled_clf.predict(X_test)

    # Fit to data and predict using pipelined scaling, GNB and PCA.
    std_clf = make_pipeline(StandardScaler(), PCA(n_components=5), GaussianNB())
    std_clf.fit(X_train, y_train)
    pred_test_std = std_clf.predict(X_test)

    # Show prediction accuracies in scaled and unscaled data.
    print('\nPrediction accuracy for the normal test dataset with PCA')
    print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))

    print('\nPrediction accuracy for the standardized test dataset with PCA')
    print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test_std)))

    # Extract PCA from pipeline
    pca = unscaled_clf.named_steps['pca']
    pca_std = std_clf.named_steps['pca']

    # Show first principal components
    print('\nPC 1 without scaling:\n', pca.components_[0])
    print('\nPC 1 with scaling:\n', pca_std.components_[0])

    # Use PCA without and with scale on X_train data for visualization.
    X_train_transformed = pca.transform(X_train)
    scaler = std_clf.named_steps['standardscaler']
    X_train_std_transformed = pca_std.transform(scaler.transform(X_train))

    # visualize standardized vs. untouched dataset with PCA performed
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,6))


    for l, c, m in zip(range(0, 3), ('blue', 'red', 'green' ), ('^', 's', 'o' )):
        ax1.scatter(X_train_transformed[y_train == l, 0],
                    X_train_transformed[y_train == l, 1],
                    color=c,
                    label='class %s' % l,
                    alpha=0.5,
                    marker=m
                    )

    for  l, c, m   in zip(range(0, 3), ('blue', 'red', 'green' ), ('^', 's', 'o' )):
        ax2.scatter(X_train_std_transformed[y_train == l, 0],
                    X_train_std_transformed[y_train == l, 1],
                    color=c,
                    label='class %s' % l,
                    alpha=0.5,
                    marker=m
                    )

    ax1.set_title('Training dataset after PCA')
    ax2.set_title('Standardized training dataset after PCA')

    for ax in (ax1, ax2):
        ax.set_xlabel('1st principal component')
        ax.set_ylabel('2nd principal component')
        ax.legend(loc='upper right')
        ax.grid()

    plt.tight_layout()

    plt.show()
 
    return 'High'

def kmeans():
    ids2 = ids.drop(['Patient_ID', 'prediction', 'Proba'], axis=1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X) 
    y = ids2['y_target']
    X = ids2.copy() 
    means = KMeans(n_clusters=2, random_state=0).fit(X)

    return None

if __name__ == "__main__":
    print(pca() )  
