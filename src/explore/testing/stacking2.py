import pandas as pd 
import numpy as np 
pd.set_option('display.max_columns', None) 
# compare ensemble to each baseline classifier
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier , RandomForestClassifier
from matplotlib import pyplot as plt   
import matplotlib.gridspec as gridspec

from mlxtend.classifier import StackingClassifier
from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions
import itertools 
import seaborn as sns

df = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/ready12_24_train.csv')  
del df['Health_Camp_ID']
from preprocessing import drop_cols , one_hot_encoding , scale , drop_cols_specific

dataframe_1 = scale(df) #Scale all features that are not binary 



def stacked(dataframe):
    y= dataframe.pop('y_target')
    X = dataframe 

    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    lr = LogisticRegression()
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], 
                            meta_classifier=lr)



    label = ['KNN', 'Random Forest', 'Naive Bayes', 'Stacking Classifier']
    clf_list = [clf1, clf2, clf3, sclf]
        
    fig = plt.figure(figsize=(10,8))
    gs = gridspec.GridSpec(2, 2)
    grid = itertools.product([0,1],repeat=2)

    clf_cv_mean = [] 
    clf_cv_std = []
    for clf, label, grd in zip(clf_list, label, grid):
            
        scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
        print (f"Accuracy: {scores.mean()} , {scores.std()} , {label}" ) 
        clf_cv_mean.append(scores.mean())
        clf_cv_std.append(scores.std())
            
    #     clf.fit(X, y)
    #     ax = plt.subplot(gs[grd[0], grd[1]])
    #     X_ = np.array(X)  
    #     y_ = y.astype(np.integer)
    #     fig = plot_decision_regions(X=X_, y=y_, clf=clf)
    #     plt.title(label)

    # plt.show()
    return clf_cv_mean ,clf_cv_std 


'''
Accuracy: 0.759225882173281 , 0.009699472463508928 , KNN
Accuracy: 0.8167726154883228 , 0.009900209751242091 , Random Forest
Accuracy: 0.7322457487743342 , 0.021004764447644184 , Naive Bayes
Accuracy: 0.8167726154883228 , 0.009900209751242091 , Stacking Classifier
<function stacked at 0x7f235f23e670>
'''




if __name__ == '__main__':
    
    df_encode= one_hot_encoding(dataframe_1, columns = ['Category 1','Category 2','Category 3','Job Type_x', 'online_score','City_Type2_x']) # for testing differnt cities dont drop City_Type
    #df_encode2= one_hot_encoding(dataframe_2, columns = ['Category 1','Category 2','Category 3','Job Type', 'online_score']) #, 'City_Type2_x'
    
    df_encode1 = df_encode.copy() 
    #df_encode4 = df_encode2.copy() 

    df_test = df_encode1.drop(['Category 1','Category 2','Category 3','Job Type_x', 'online_score','City_Type2_x'],axis=1) #,'City_Type2_x'
    df_test1 = df_test.copy()  
   
 
    stacked_output = stacked(df_test1)
    print(stacked)