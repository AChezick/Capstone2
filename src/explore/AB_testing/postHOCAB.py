
# 5/10/21
'''
AB_testing tab is currently under construction.

by_date_camps.py is the main testing file
postHOCAB.py is the modeling file 



This script :
1. Takes in train DF to train (which is all patients & health camps ending before test_df starts)
2. Each model's probabilities for patient attendance will be added as columns to the test_DF
3. Test DF will be sent back for parsing in AB testing. 
'''



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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
#from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV
# from preprocessing import drop_cols , one_hot_encoding , scale   #5/20 trying to use the pre_format file
import xgboost as xgb   
from xgboost import XGBClassifier 
from sklearn.metrics import mean_squared_error

from preprocessing_formatting import drop_cols , drop_cols_specific, one_hot_encoding , scale , scale2    
#from city_testing import run_test_typeC
import itertools as it 
dataframe = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/temp_csv/ab_df.csv')
# deleting columns that mess with svc - which are words.
# del dataframe['Job Type_x']
# del dataframe['Category 1']
# del dataframe['Category 2']
# del dataframe['Category 3']
# del dataframe['Start']
# del dataframe['End']

def temp_test():
    '''
    Creating test function to get stuff working
    '''
    ans ={}
    df_encode1 = dataframe.drop(['City_Type2_x','Job Type_x','Category 2','Category 3','Category 1', 'online_score', 'Start','End'],axis=1) 
    df1,df2 = df_encode1.copy() , df_encode1.copy() 


    for index,item in enumerate([(6544, ['NA', 6530, 6560]) , (6561, ['NA', 6530, 6560, 6544]), (6585, ['NA', 6530, 6560, 6544])]): 
        if len(item[1]) <=1: #
            break
        else:
            iD = item[0] # Camp_ID
            camps = item[1][1:] #list of camps for training 

            test_df = df1[df1['Health_Camp_ID'] == iD ]  
            train_df = df2.loc[ df2['Health_Camp_ID'].isin(camps)  ]  

            print(train_df , 'TRAIN', test_df)

            get_result = run_tests(test_df,train_df)
            ans[index] = get_result
        #5/20 need to create new cols for ans / figure out how to record results 

    return  ans


def run_tests(test_df,train_df ):
    '''
    get dfs, make copies, run tests , combine_results, send back for AB testing
    '''
    final_df = test_df.copy() 
    test_df1 , train_df1 = test_df.copy() , train_df.copy() 
    test_df2 , train_df2 = test_df.copy() , train_df.copy() 
    test_df3 , train_df3 = test_df.copy() , train_df.copy() 

    get_L =  run_test_typeL(test_df1 , train_df1)
    get_knn = run_test_typek(test_df2 , train_df2)
    get_svc = run_test_typeS(test_df3 , train_df3)
    
    # combine results
    final_df['SVC'] = get_svc['prediction'].values
    final_df['svc_preds'] = get_svc['proba']
    final_df['XG'] = get_knn['prediction']
    final_df['xg_preds'] = get_knn['Proba']
    final_df['log'] = get_L['predictionL']
    final_df['log_preds'] = get_L['probaL']

    return final_df

def run_test_typeAA(test_df1 , train_df1):
    '''
    run an XDG Classifier  
    -No splitting needed.
    Train needs no split but does need a target column
    Test needs no split but does need a target column 
    '''
    #y = target || X =  data_frames 
   
    y_testD , y_trainD = test_df1.pop('y_target'),  train_df1.pop('y_target')
    X_testD , X_trainD = test_df1, train_df1
 
    del X_testD['Health_Camp_ID'] 
    del X_testD['Patient_ID'] 
    #del y_testD['Health_Camp_ID']  
    #del y_testD['Patient_ID'] 
     
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
                max_depth = 12, alpha = 8, n_estimators = 12, eval_metric = 'auc', label_encoder=False,scale_pos_weight=2)

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

def run_test_typeS(test_dfs , train_dfs):  
    '''
    Runs SVC, adds back columns 
    train_y = train_df1.pop('y_target')
    test_y = test_df1.pop('y_target')

    df_train = train_df1
    df_test = test_df1
    '''
    trainz_y = train_dfs['y_target'].values
    testz_y = test_dfs['y_target'].values 

    test_dfs = scale2(test_dfs)
    train_dfs = scale2(train_dfs)



    del train_dfs['y_target']
    del test_dfs['y_target']

    train_y = trainz_y
    test_y = testz_y

    df_train = train_dfs
    df_test = test_dfs
    

    #might need to delete / edit DF
    del df_train['Health_Camp_ID'] 
    del df_test['Health_Camp_ID']  

    del df_train['Patient_ID'] 
    del df_test['Patient_ID'] 
    # print(df_train.columns,df_test.columns, 'next are the y', train_y,test_y)
    #X_trainS, X_testS, y_trainS, y_testS = train_test_split(x, y, test_size=0.2, random_state=101) 
    #^ the above line wont be needed 

    svc = SVC(random_state=101 ,probability=True)
    svc.fit(df_train,train_y)
    svc_preds = svc.predict(df_test)
    svc_proba = svc.predict_proba(df_test)[:,1]

    predsx , preds2x = svc_proba  >= .5 , svc_proba  >= .3

    df_test['prediction'] = svc_preds
    df_test['proba'] = svc_proba
    df_test['y_target'] = test_y

    return df_test 

def run_test_typek(test_dfk , train_dfk):
    '''
    run knn for post hoc- analysis 
    '''
    trainz_y = train_dfk['y_target'].values #getting y_target values for test & train
    testz_y = test_dfk['y_target'].values 

    del train_dfk['y_target'] #deleting values 
    del test_dfk['y_target']

    train_y = trainz_y 
    test_y = testz_y

    df_train = train_dfk
    df_test = test_dfk
    

    #might need to delete / edit DF
    del df_train['Health_Camp_ID'] 
    del df_test['Health_Camp_ID']  

    del df_train['Patient_ID'] 
    del df_test['Patient_ID'] 

    #print('train ---->', train_dfk, train_y) # ensuring both dfs print
    #print('test ---->', test_dfk , test_y)

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(df_train,train_y)
    knn_preds = knn.predict(df_test)
    knn_proba = knn.predict_proba(df_test) 
     
    print(knn_proba, 'is proba'  )
    print(knn_preds, 'is preds'  )

    df_test['prediction'] = knn_preds
    df_test['Proba'] = knn_proba[ :,1]   
    df_test['y_target'] = test_y

    return df_test 

def run_test_typeL(test_dfl , train_dfl):
    '''
    run knn for post hoc- analysis 
    '''
    train_y = train_dfl['y_target'].values #getting y_target values for test & train
    test_y = test_dfl['y_target'].values 

    del train_dfl['y_target'] #deleting values 
    del test_dfl['y_target']

    df_train = train_dfl
    df_test = test_dfl
    
    del df_train['Health_Camp_ID'] 
    del df_test['Health_Camp_ID']  

    del df_train['Patient_ID'] 
    del df_test['Patient_ID'] 
    
    w = {0:59, 1:41} 
    logmodelx = LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=1.0, 
            fit_intercept=True, intercept_scaling=1, class_weight=w , random_state=None, 
            solver='lbfgs', max_iter=50, multi_class='auto', verbose=0, warm_start=False, 
            n_jobs=1, l1_ratio=None ) 
    logmodelx.fit(df_train, train_y)

    pure_probaz = logmodelx.predict_proba(df_test) 
    predictionsz = logmodelx.predict(df_test)

    df_test['predictionL'] = pure_probaz[:,-1]  #logmodelx.predict(df_test)
    df_test['probaL'] =  predictionsz   #logmodelx.predict_proba(df_test)[:,-1]   

    return df_test 

if __name__ =='__main__':
    print(temp_test())

'''
5/6
input will be test_df, train_df 
1. send to run_tests
-makes copies of DF
-sends a copy to each model
1b. gets back DF with proba,predict rows
-add rows from each model to the test_df copy
1c. Send back test_df with all bandit/model results 

5/10
---To help with debugging I have inserted a temp_testing function
5/12
--Next step is to do hot encoding of text features, also will need to change names of columns for xg boosts 
--Knn model is working , mostly ... 
5/13
-- functions not importing from other python files (for scaling, dropping, etc)
-- this is preventing the models from working
5/20
-- Test function is working, scaling is working, dropping is working
-- SVC test is working, KNN is not, not sure on XG
-- Next step is to try and pass dfs from by_date_camps
'''
 

