import pandas as pd 
import numpy as np 
from numpy import argmax 
pd.set_option('display.max_columns', None) 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix, classification
from sklearn.model_selection import train_test_split ,  RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.metrics import plot_roc_curve

#from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
import xgboost as xgb
from xgboost import XGBClassifier 
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/ready12_24_train.csv')  

from preprocessing import drop_cols , one_hot_encoding , scale
data = drop_cols(df) # drop cols
dataframe_1 = scale(data) #Scale all features that are not binary 

import matplotlib.pyplot as plt
from statsmodels.tools import add_constant
from scipy.stats import uniform 

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

def run_test_typeA(X_testz,y_trainz): # this test might have to be done differently than the other two for the holdout aspect 
    '''
    Remove columns, print list of columns, drop columns, run model
    ''' 
    y = y_trainz
    x = X_testz
    #data_dmatrix = xgb.DMatrix(data=x,label=y) 
    X_traina, X_testa, y_traina, y_testa = train_test_split(x, y, test_size=0.2, random_state=101) 
    
    svc = SVC(random_state=42, probability=True)
    svc.fit(X_traina, y_traina)
    svc_preds = svc.predict(X_testa)
    svc_proba = svc.predict_proba(X_testa)[:,1]

    predsx , preds2x = svc_proba  >= .5 , svc_proba  >= .4

    print(classification_report(y_testa,svc_preds) )
    svc_disp = plot_roc_curve(svc, X_testa, y_testa)
    plt.show()

    return None 

def run_test_forest(X_testz,y_trainz): # THIS NEEDS TO be split for train and test somehow .
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



def run_test_boost(X_testz,y_trainz):
    '''
    run an XDG Classifier  
    '''
    g_classifier =GradientBoostingClassifier() 
    y = y_trainz
    x = X_testz

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
    plot = plot_partial_dependence(final_model, features=[5,6,7],feature_names = ['delta_first_reg',
           'delta_first_start', 'delta_reg_end'], X = X_holdout)
    return (f'for {Classifier} the final model with optimized hyperparameters results in {dict_}', dict_.items())

def graph_or_table():
    '''
    construct a graph or nice format from test_2  
    '''

if __name__ == '__main__':

    df_encode = one_hot_encoding(dataframe_1, columns = ['City_Type','Category1_x','Category2','Category3','Job_Type', 'online_score'])
    df_encode1 = df_encode.copy() 
    df_test = df_encode1.drop(['City_Type','Category1_x','Category2','Category3','Job_Type', 'online_score'],axis=1)
    df_test1, df_test2 = df_test.copy() , df_test.copy()

    X_train, X_holdout, y_train, y_holdout= create_holdout(df_test) 

    # randomforest_bestparams , randomforest_bestscore = run_test_forest(X_train, y_train)
    # randomforest_finalmodel = final_test(X_train, y_train, X_holdout, y_holdout,RandomForestClassifier,randomforest_bestparams)
 
    # gboost_bestparams,gboost_bestscore = run_test_boost(X_train, y_train)
    # gboosting_final = final_test(X_train, y_train, X_holdout, y_holdout, GradientBoostingClassifier, gboost_bestparams)

    log_bestparams, log_bestscore = run_log_reg(X_train, y_train)
    log_finalmodel = final_test(X_train, y_train, X_holdout, y_holdout, LogisticRegression , log_bestparams )
    print(log_bestparams, log_bestscore   ,log_finalmodel )
     


    # print(run_test_typeA(X_train,y_train))
    #print(run_test_typeB(X_train,y_train))
    #print(run_test_typeD(X_train,y_train))

    
 
 
            

     
    
