import pandas as pd 
import numpy as np 
from numpy import argmax 
pd.set_option('display.max_columns', None) 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix, classification
from sklearn.model_selection import train_test_split ,  RandomizedSearchCV,ParameterGrid,GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.metrics import plot_roc_curve
from sklearn.neighbors import KNeighborsClassifier

#from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV
from sklearn.inspection import plot_partial_dependence 
import xgboost as xgb
from xgboost import XGBClassifier     
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/ready12_24_train.csv')  

# df_plain = pd.read_csv('/home/allen/Galva/capstones/capstone2/Notebooks/plain_df_to_test.csv')

# del df_no_scale['Health_Camp_ID']
# del df_no_scale['Category1_x']
# del df_no_scale['Category2']

from preprocessing import drop_cols , one_hot_encoding , scale
data = drop_cols(df) # drop cols
dataframe_1 = scale(data) #Scale all features that are not binary  

import matplotlib.pyplot as plt
from statsmodels.tools import add_constant
from scipy.stats import uniform 
#print(df_no_scale.head(10))
def create_holdout(dataframe ):
    '''
    Create holdout dataframe 
    '''
    dataframe.y_target = dataframe.y_target.astype(int)
    y = dataframe.pop('y_target') 
    X = dataframe 
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, 
                                                        shuffle=False,
                                                        test_size=0.3, 
                                                        random_state=101)

    return X_train, X_holdout, y_train, y_holdout


def run_test_forest(X_train,y_train): # THIS NEEDS TO be split for train and test somehow .
    '''
    run the test
    '''
    classifier = RandomForestClassifier()
    # parameters to investigate
    num_estimators = [90,150,300,1000]
    criterion = ["gini", 'entropy']
    max_features = ['auto', 'sqrt', "log2"]
    max_depth = [10, 20, None]
    min_samples_split = [2, 6, 10]
    min_samples_leaf = [3, 5,7]
    # dictionary containing the parameters
    param_grid = {'n_estimators': num_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'criterion': criterion}
    # hyperparameter tuning parameter search methodology
    random_search = RandomizedSearchCV(classifier, 
                                   param_grid, 
                                   scoring='accuracy',
                                   cv=3,
                                   n_iter=200,
                                   verbose=1,
                                   return_train_score=True,
                                   n_jobs=-1)
    random_search.fit(X_train, y_train)
    # print the bestparameters and the best recall score
    randomforest_bestparams = random_search.best_params_
    randomforest_bestscore =  random_search.best_score_
    return randomforest_bestparams, randomforest_bestscore 

def run_test_boost(X_train,y_train):
    '''
    run an XDG Classifier  
    '''
    g_classifier =GradientBoostingClassifier() 
    y = y_train
    x = X_train

    num_estimators = [20,50,100,200]
    criterion = ['friedman_mse', 'mse']
    max_features = ['auto', 'sqrt', "log2"]
    max_depth = [4, 8, None]
    min_samples_split = [2, 10, 30]
    min_samples_leaf = [1, 3]

    param_grid = {'n_estimators': num_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'criterion': criterion}

    g_random_search = RandomizedSearchCV(g_classifier, 
                                   param_grid, 
                                   scoring='accuracy',
                                   cv=3,
                                   n_iter=200,
                                   verbose=1,
                                   return_train_score=True,
                                   n_jobs=-1)
    g_random_search.fit(X_train, y_train)
    # print the bestparameters and the best recall score
    gradientboosting_bestparams = g_random_search.best_params_
    gradientboosting_bestscore =  g_random_search.best_score_
    return gradientboosting_bestparams , gradientboosting_bestscore

def run_log_reg(X_testz,y_trainz,solver=['liblinear']):
    '''
    do part 2 _ maybe optomize  ? 
    '''
    logistic = LogisticRegression()
    # parameters to investigate
    if solver == ['lbfgs'] or solver == ['newton-cg'] or solver== ['sag']:
        penalty = ['l2', 'none']
    elif solver == ['liblinear']:
        penalty = ['l1', 'l2'] 
    elif solver == ['sage']:
        penalty = ['elasticnet']
    distributions = dict(C=uniform(loc=0, scale=4),
                    penalty=penalty,
                    max_iter= [100, 200, 500, 1000],
                    solver= solver)
    # hyperparameter tuning parameter search methodology
    desired_iterations = 100
    search = RandomizedSearchCV(logistic, 
                                   distributions, 
                                   scoring='recall',
                                   cv=3,
                                   n_iter=desired_iterations,
                                   verbose=1,
                                   return_train_score=True,
                                   n_jobs=-1, 
                                   random_state=0)
    log_search = search.fit(X_train, y_train)
    log_bestparams = log_search.best_params_
    log_bestscore =  log_search.best_score_
    return log_bestparams, log_bestscore 


def run_test_knn(df_encode1):  
    '''
    run the test
    '''
    # classifier = KNeighborsClassifier()
    # # parameters to investigate
 
    # # dictionary containing the parameters
    # param_grid = {'n_neighbors': [9,10,11],
    #           'weights':['uniform','distance'],
    #           'metric': ['euclidean','manhattan']}
    # # hyperparameter tuning parameter search methodology
    # gs = GridSearchCV(classifier, 
    #                                param_grid, 
    #                                scoring='accuracy',
    #                                cv=3,
    #                                n_iter=200,
    #                                verbose=1,
    #                                return_train_score=True,
    #                                n_jobs=-1)
    # gs_results = gs.fit(X_train, y_train)
    # # print the bestparameters and the best recall score
    # knn_bestparams = gs_results.best_params_
    # knn_bestscore =  gs_results.best_score_ 
    yk = df_encode1.pop('y_target')
    Xk = df_encode1 

    X_traink, X_testk, y_traink, y_testk = train_test_split(Xk, yk, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_traink,y_traink)
    pred = knn.predict(X_testk)
    cm = confusion_matrix(y_testk,pred)
    cr = classification_report(y_testk,pred)
    return cm , cr  

def final_test(X_train, y_train, X_holdout, y_holdout,Classifier,kwargs = {}):
    '''
    Execute trained model on holdout data, return results 
    '''
    final_model= Classifier(**kwargs)
    final_model.fit(X_train, y_train)
 
    y_predict= final_model.predict(X_holdout)
    score=final_model.score(X_holdout, y_holdout)
    precision=precision_score(y_holdout, y_predict)
    recall=recall_score(y_holdout, y_predict)
    conf_matrix = confusion_matrix(y_holdout, y_predict)
    dict_={"score": score, "precision": precision, 'recall':recall, 'confusion_matrix': conf_matrix}
    # features=[5,6,7],feature_names = ['delta_first_reg','delta_first_start', 'delta_reg_end'], X = X_holdout)
    # features = [10,11,12] 
    # plot = plot_partial_dependence(final_model,X_holdout,features) 
    # plt.show()  
    # print(plot)
    return (f'for {Classifier} the final model with optimized hyperparameters results in {dict_}', dict_.items())

def graph_or_table():
    '''
    construct a graph or nice format from test_2  
    '''

if __name__ == '__main__':
     

    df_encode = one_hot_encoding(dataframe_1, columns = ['City_Type2_x','Category 1','Category 2','Category 3','Job Type_x', 'online_score'])
    df_encode1 = df_encode.drop(['City_Type2_x','Category 1','Category 2','Category 3','Job Type_x','online_score'],axis=1) 
    del df_encode['Health_Camp_ID'] 
    # #df_encode1.to_csv('/home/allen/Galva/capstones/capstone2/data/hot_drop_21.csv')  
    df_encode1 = df_encode1.drop([1,2,3,4,'Camp Start Date - Registration Date',
       'Registration Date - First Interaction',
       'Camp Start Date - First Interaction',
       'Camp End Date - Registration Date', 'Camp Length'],axis=1) 
     
    df_test1, df_test2 = df_encode1.copy() , df_encode1.copy()
    print(df_test1)
    X_train, X_holdout, y_train, y_holdout= create_holdout(df_encode1) 
     
    randomforest_bestparams , randomforest_bestscore = run_test_forest(X_train, y_train)
    randomforest_finalmodel = final_test(X_train, y_train, X_holdout, y_holdout,RandomForestClassifier,randomforest_bestparams)
 
    # gboost_bestparams,gboost_bestscore = run_test_boost(X_train, y_train)
    # gboosting_final = final_test(X_train, y_train, X_holdout, y_holdout, GradientBoostingClassifier, gboost_bestparams)

    # # log_bestparams, log_bestscore = run_log_reg(X_train, y_train)
    # # log_finalmodel = final_test(X_train, y_train, X_holdout, y_holdout, LogisticRegression , log_bestparams )

    #knn_bestscore, knn_best_params = run_test_knn(df_encode1)
    #knn_finalmodel = final_test(X_train, y_train, X_holdout, y_holdout, KNeighborsClassifier , knn_bestparams )

    print(randomforest_bestparams , randomforest_finalmodel )
      
 


            
 