from sklearn import (cluster, datasets, decomposition, ensemble, manifold, random_projection, preprocessing)
import numpy as np
import pandas as pd 

from matplotlib import cm
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, plot_roc_curve, accuracy_score

pd.set_option('display.max_columns', None) 

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from pandas.plotting import scatter_matrix

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from utils import XyScaler
from roc_curve2 import roc_curve
from scipy.stats import uniform 
###########################################################################################33
'''
train_set = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/train_1.csv')
test_set = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/test_df1.csv') 

train1 = train_set.drop(['Unnamed: 0.1.1','Unnamed: 0', 'Unnamed: 0.1','Registration_Date', 
'Employer_Category','Patient_ID','First_Interaction', 'Health_Camp_ID','Income',
'Education_Score','Age', 'City_Type', 'Job_Type', 'Education_Score2',
'Education_Scorez',],axis=1) 
#train1 = train1.drop('Unnamed: 0.1.1.1')
train1.drop(train1.columns[5], axis=1, inplace=True) 

test_set = test_set.drop(['Unnamed: 0.1.1','Unnamed: 0', 'Unnamed: 0.1','Registration_Date', 
'Employer_Category','Patient_ID','First_Interaction', 'Health_Camp_ID','Income',
'Education_Score','Age', 'City_Type', 'Job_Type', 'Education_Score2',
'Education_Scorez',],axis=1) 
test_set.drop(test_set.columns[5], axis=1, inplace=True) 

y1, y2 = train1.pop('Event1_or_2') , test_set.pop('Event1_or_2')
'''
###########################################################################################33




checker = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/train_4_model.csv')
checker1 = checker.copy() 

from sklearn.model_selection import train_test_split 
ynot = checker.pop('y_target')
train_test_split(ynot, shuffle=False) 
X_train,X_test,y_train,y_test = train_test_split(
    checker,ynot,test_size = .3, random_state=101)


pca = decomposition.PCA(n_components=8)
X_pca = pca.fit_transform(X_test)

def scree_plot(ax, pca, n_components_to_plot=7, title=None): 

    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    ax.plot(ind, vals, color='blue')
    ax.scatter(ind, vals, color='blue', s=50)

    for i in range(num_components):
        ax.annotate(r"{:2.2f}%".format(vals[i]), 
                   (ind[i]+0.2, vals[i]+0.005), 
                   va="bottom", 
                   ha="center", 
                   fontsize=12)

    ax.set_xticklabels(ind, fontsize=12)
    ax.set_ylim(0, max(vals) + 0.05)
    ax.set_xlim(0 - 0.45, n_components_to_plot + 0.45)
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=16)


fig, ax = plt.subplots(figsize=(10, 6))
scree_plot(ax, pca, title="Principal Components for Attendance")
plt.show() 



'''
  File "pca_cap2.py", line 97, in <module>
    importances = X_pca.feature_importances_
AttributeError: 'numpy.ndarray' object has no attribute 'feature_importances_'
''' # Error for trying below : 
# def cm_to_inch(value):
#     return value/2.54

# col_names = X_test.columns 
# importances = X_pca.feature_importances_
# indices = np.argsort(importances)[::-1]
# Random_Forest = 'Random Forest'
# plt.bar(range(X_test.shape[1]), importances[indices], color="b")
# plt.title("{} Feature Importances".format(Random_Forest))
# plt.xlabel("Feature")
# plt.ylabel("Feature importance")
# plt.xticks(range(X_test.shape[1]), col_names[indices], rotation=45, fontsize=12, ha='right')
# plt.xlim([-1, X_test.shape[1]])
# plt.figure(figsize=(cm_to_inch(15),cm_to_inch(10)))