import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
plt.rcParams['figure.dpi'] = 200 
train_set = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/train_1.csv')
test_set = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/test_df1.csv') 

print(train_set.columns, test_set.columns)

train1 = train_set.drop(['Unnamed: 0.1.1','Unnamed: 0', 'Unnamed: 0.1', 'Registration_Date','Employer_Category','Patient_ID','First_Interaction', 'Health_Camp_ID','Income',
'Education_Score','Age', 'City_Type', 'Job_Type', 'Education_Score2',
'Education_Scorez',],axis=1) 
#train1 = train1.drop('Unnamed: 0.1.1.1')
#train1.drop(train1.columns[['5','12']], axis=1, inplace=True) 

'''
['Employer_Category' 'First_Interaction' 
'Income' 'Education_Score' 'Age'\n 'City_Type' 'Job_Type' 'Education_Score2' 'Education_Scorez'] not found in axis"



'''



test_set = test_set.drop(['Unnamed: 0.1.1','Unnamed: 0', 'Unnamed: 0.1','Registration_Date','Employer_Category','Patient_ID','First_Interaction', 'Health_Camp_ID','Income',
'Education_Score','Age', 'City_Type', 'Job_Type', 'Education_Score2',
'Education_Scorez',],axis=1) 
#test_set.drop(test_set.columns['5','12'], axis=1, inplace=True) 

y1, y2 = train1.pop('Event1_or_2') , test_set.pop('Event1_or_2')

# Define a pipeline to search for the best combination of PCA truncation
# and classifier regularization.
pca = PCA()
# set the tolerance to a large value to make the example faster
logistic = LogisticRegression(max_iter=10000, tol=0.1)
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)]) 

param_grid = {'pca__n_components': [1,2,3,4,5,10,20], 'logistic__C': np.logspace(-4, 4, 4),} 
search = GridSearchCV(pipe, param_grid, n_jobs=-1)
search.fit(test_set, y2)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

pca.fit(test_set)
fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
ax0.plot(np.arange(1, pca.n_components_ + 1),
         pca.explained_variance_ratio_, '+', linewidth=2)
ax0.set_ylabel('PCA explained variance ratio')

ax0.axvline(search.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
ax0.legend(prop=dict(size=12))

results = pd.DataFrame(search.cv_results_)
components_col = 'param_pca__n_components'
best_clfs = results.groupby(components_col).apply(
    lambda g: g.nlargest(1, 'mean_test_score'))

best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
               legend=False, ax=ax1)
ax1.set_ylabel('Classification accuracy (val)')
ax1.set_xlabel('n_components')

plt.xlim(-1, 70)

plt.tight_layout()
plt.show()  

def plot_mnist_embedding(ax, X, y, title=None):
    """Plot an embedding of the mnist dataset onto a plane.
    
    Parameters
    ----------
    ax: matplotlib.axis object
      The axis to make the scree plot on.
      
    X: numpy.array, shape (n, 2)
      A two dimensional array containing the coordinates of the embedding.
      
    y: numpy.array
      The labels of the datapoints.  Should be digits.
      
    title: str
      A title for the plot.
    """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    ax.axis('off')
    ax.patch.set_visible(False)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], 
                 y , 
                 color=plt.cm.Set1(y[i] / 10.), 
                 fontdict={'weight': 'bold', 'size': 12})

    ax.set_xticks([]), 
    ax.set_yticks([])
    ax.set_ylim([-0.1,1.1])
    ax.set_xlim([-0.1,1.1])

    if title is not None:
        ax.set_title(title, fontsize=16)

pca = decomposition.PCA(n_components=5)
X_pca = pca.fit_transform(test_set)


fig, ax = plt.subplots(figsize=(10, 6))
plot_mnist_embedding(ax, X_pca, y2)    