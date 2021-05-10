import pandas as pd 
import numpy as np 
pd.set_option('display.max_columns', None) 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
#from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV
  
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from preprocessing import drop_cols , one_hot_encoding , scale
from city_testing import run_test_typeC
import itertools as it 
df_no_scale = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/temp_csv/hot_code_NO_scale.csv')
ids = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Train.csv')
#del df_no_scale['Health_Camp_ID']
del df_no_scale['Category1_x']
del df_no_scale['Category2']
del df_no_scale['Category3']
del df_no_scale['City_Type']
del df_no_scale['Job_Type']
del df_no_scale['online_score']

hot_scale = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/temp_csv/train_4_model.csv')

def run_tests(test_df,train_df ):
    '''
    get dfs, make copies, run tests , combine_results, send back for AB testing
    '''
    final_df = test_df.copy() 
    #test_df1 , train_df1 = test_df.copy() , train_df.copy() 
    #test_df2 , train_df2 = test_df.copy() , train_df.copy() 
    #test_df3 , train_df3 = test_df.copy() , train_df.copy() 

    get_xg =  run_test_typeAA(test_df1 , train_df1)
    get_knn = run_test_typek(test_df2 , train_df2)
    get_svc = run_test_typeS(test_df3 , train_df3)
    
    # combine results
    

    return final_df

def run_test_typeAA(test_df1 , train_df1):
    '''
    run an XDG Classifier  
    -No splitting needed.
    Train needs no split but does need a target column
    Test needs no split but does need a target column 

    '''
    #y = target
    y_testD , y_trainD = test_df1.pop('y_target'),  train_df1.pop('y_target')
    X_testD , X_trainD = test_df1, train_df1
 
    del X_testD['Health_Camp_ID'] 
    del X_testD['Patient_ID'] 
    del y_testD['Health_Camp_ID']  
    del y_testD['Patient_ID'] 
     
    #For Post-Hoc Tracking
    # X_trainIDs = X_trainD.loc['Patient_ID']
    # X_testIDs = X_testD.loc['Patient_ID'] 
    # y_trainIDs = y_trainD.loc['Patient_ID']
    # y_testIDs =  y_testD.loc['Patient_ID']
    # del X_trainD['Patient_ID']
    # del X_testD['Patient_ID']
    # del y_trainD['Patient_ID'] 
    # del y_testD['Patient_ID']
 
    xg_reg1 = XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 8, alpha = 8, n_estimators = 12, eval_metric = 'auc', label_encoder=False,scale_pos_weight=2)

    xg_reg1.fit(X_trainD, y_trainD)  

    xg_reg1_predict = xg_reg1.predict(X_testD ) #(0/1 associated with .5)
    xg_reg1_proba = xg_reg1.predict_proba(X_testD )[:,1]
     
    preds_xg1_thresh1 =xg_reg1_proba>=0.5
    preds2_xg1_thresh2 = xg_reg1_proba>=0.35

    mat1 = confusion_matrix(y_testD,preds_xg1_thresh1 ) 
    mat2 = confusion_matrix(y_testD,preds2_xg1_thresh2 ) 

    y_counts_test = sum([1 for x in y_testD.values if x ==1])
    y_counts_train = sum([1 for x in y_trainD.values if x ==1])

    conf_matrix1,conf_matrix2  = [],[] 
    test_size =len(X_testD)
    train_size =len(X_trainD)

    X_testD['prediction'] = xg_reg1_predict
    X_testD['Proba'] = xg_reg1_proba
    X_testD['y_target'] = y_testD 

    return X_testD 

def run_test_typeS(test_df1 , train_df1):  
    '''
    Runs SVC, adds back columns 
    '''
    y_testS, y_trainS = test_df1.pop('y_target'),  train_df1.pop('y_target')
    X_testS, X_trainS = test_df1, train_df1


    #might need to delete / edit DF
    # del X_testD['Health_Camp_ID'] 
    # del X_testD['Patient_ID'] 
    # del y_testD['Health_Camp_ID']  
    # del y_testD['Patient_ID'] 
  
    #X_trainS, X_testS, y_trainS, y_testS = train_test_split(x, y, test_size=0.2, random_state=101) 
    #^ the above line wont be needed 

    svc = SVC(random_state=101 ,probability=True)
    svc.fit(X_trainS, y_trainS)
    svc_preds = svc.predict(X_testS)
    svc_proba = svc.predict_proba(X_testS)[:,1]

    predsx , preds2x = svc_proba  >= .5 , svc_proba  >= .4

    print(classification_report(y_testS,svc_preds) )
    svc_disp = plot_roc_curve(svc, X_testS, y_testS)
    plt.show()

    X_testS['prediction'] = svc_preds
    X_testS['Proba'] = svc_proba
    X_testS['y_target'] = y_testS

    return X_testS 

def run_test_typek(test_df1 , train_df1):
    '''
    run knn for post hoc- analysis 
    '''
    
    y_test, y_train = test_df1.pop('y_target'),  train_df1.pop('y_target')
    X_test, X_train = test_df1, train_df1 

    X_traink, X_testk, y_traink,y_testk = train_test_split(
    X, y , test_size = 0.2, random_state=101)

    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_traink,y_traink)
    knn_preds = knn.predict(X_testk)
    knn_proba = knn.predict_proba(X_testk)[:,-1]
 
    print(confusion_matrix(y_testk,knn_preds))
    print(classification_report(y_testk,knn_preds))
    
    confs_output = confusion_matrix(y_testk,knn_preds ) 

    X_testk['prediction'] = knn_preds
    X_testk['Proba'] = knn_proba  
    X_testk['y_target'] = y_testk

    return confs_output  

if __name__ =='__main__':
    get = run_tests(test_df,train_df) #


'''
5/6
input will be test_df, train_df 
1. send to run_tests
-makes copies of DF
-sends a copy to each model
1b. gets back DF with proba,predict rows
-add rows from each model to the test_df copy
1c. Send back test_df 
'''
 

