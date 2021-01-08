import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)  
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
plt.rcParams['figure.dpi'] = 200 

checker = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/train_4_model.csv')
checker1 = checker.copy() 

from sklearn.model_selection import train_test_split 
ynot = checker.pop('y_target')
train_test_split(ynot, shuffle=False) 
X_train,X_test,y_train,y_test = train_test_split(
    checker,ynot,test_size = .3, random_state=101)


pca = decomposition.PCA(n_components=8)
X_pca = pca.fit_transform(X_test)
  
y1, y2 = y_train,y_test
print(X_test.columns) 
# Define a pipeline to search for the best combination of PCA truncation
# and classifier regularization.
pca = PCA()
# set the tolerance to a large value to make the example faster
logistic = LogisticRegression(max_iter=10000, tol=0.1)
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)]) 

param_grid = {'pca__n_components': [1,2,3,4,5,10,13], 'logistic__C': np.logspace(-4, 4, 4),} 
search = GridSearchCV(pipe, param_grid, n_jobs=-1)
search.fit(X_test, y2)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

pca.fit(X_test)
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
from sklearn import preprocessing 

len(X_test.index)

def pca(df): 
  x= df
  print(df.describe())
  pca_ = PCA(n_components=2)
  principle_components = pca_.fit_transform(x)
  principle_df = pd.DataFrame(data= principle_components
                                , columns = ['PCA1','PCA2'])
  
  features = []
  for i in range(35):
    features.append(f"{i}")
  top5=pca_.components_[0].argsort()[-5:]

  print( np.array(features)[top5] ) 
  return len(features)
 

print(pca(X_test))


 
 



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
X_pca = pca.fit_transform(X_test)


fig, ax = plt.subplots(figsize=(10, 6))
plot_mnist_embedding(ax, X_pca, y2)    