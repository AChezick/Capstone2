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

from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve

#from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV
 
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/ready12_24_train.csv')  

from preprocessing import drop_cols , one_hot_encoding , scale
data = drop_cols(df) # drop cols
dataframe_1 = scale(data) #Scale all features that are not binary 

import matplotlib.pyplot as plt

df_encode = one_hot_encoding(dataframe_1, columns = ['Category1_x','Category2','Category3','Job_Type', 'online_score'])

df_test = df_encode.drop(['Category1_x','Category2','Category3','Job_Type', 'online_score'],axis=1)

def create_(dataframe ):
    '''
    Create holdout dataframe 
    '''

    return None 

def run_test_typeA(dataframe): # this test might have to be done differently than the other two for the holdout aspect 
    '''
    Remove columns, print list of columns, drop columns, run model
    '''
    y = dataframe.pop('y_target')
    x = dataframe
    data_dmatrix = xgb.DMatrix(data=x,label=y) 
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=101) 
    
    svc = SVC(random_state=42)
    svc.fit(X_test, y_test)
    svc_preds = svc.predict(X_test)

    print(classification_report(y_test,svc_preds) )
    svc_disp = plot_roc_curve(svc, X_test, y_test)
    plt.show()

    return None 

def run_test_typeB(dataframe):  
    '''
    run the test
    '''
    y = dataframe.pop('y_target')
    x = dataframe

    X_trainD, X_testD, y_trainD, y_testD = train_test_split(x, y, test_size=0.2, random_state=101) 
     
    data_dmatrix_train = xgb.DMatrix(data=X_trainD,label=y_trainD) # not sure what these are doing ?! 
    data_dmatrix_test = xgb.DMatrix(data=X_testD,label=y_testD) 

    xg_reg1 = XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 6, alpha = 8, n_estimators = 12, eval_metric = 'auc', label_encoder=False)

    xg_reg1.fit(X_trainD,y_trainD) 

    xg_reg1_predict = xg_reg1.predict(X_testD) #(0/1 associated with .5)
    xg_reg1_proba = xg_reg1.predict_proba(X_testD)[:,1]
    #print(xg_reg1_proba, 'this is the predict proba')
    preds_xg1_thresh1, preds2_xg1_thresh2 = xg_reg1_proba>=0.35 , xg_reg1_proba>=0.45


    # xg_reg1_disp = plot_roc_curve(xg_reg1 , X_testD, y_testD)
    # plt.show()

    print(classification_report(y_testD,xg_reg1_predict ) )
    print(confusion_matrix(y_testD,preds_xg1_thresh1))  

    print(confusion_matrix(y_testD,preds2_xg1_thresh2))  



def run_test_typeC(dataframe):
    '''
    do part 2 _ maybe optomize  ? 
    '''
    y = dataframe.pop('y_target')
    X = dataframe 
    X_train, X_test, y_train,y_test = train_test_split(
    X, y , test_size = 0.2, random_state=101)

    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)
    LogisticRegression(penalty='l1', dual=False, C=1.0, 
        fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, 
        solver='saga', max_iter=None , multi_class='auto', verbose=0, warm_start=False, 
        n_jobs=-1, l1_ratio=None) 
    
    predictions = logmodel.predict(X_test)
    proba = logmodel.predict_proba(X_test)[:,1]
    
    log_thresh1, log_thresh2, log_thresh3 = proba>=0.35 , proba>=0.45 , proba>=.55

    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test,log_thresh1 ), 'thresh1')
    print(confusion_matrix(y_test,log_thresh2 ), 'THRESH2*******************')
    print(confusion_matrix(y_test,log_thresh3 ), '3333')
    return None 

def final_test(X_train, y_train, X_holdout, y_holdout,Classifier, **kwargs):
    '''
    Execute trained model on holdout data, return results 
    '''
    return None 

def graph_or_table():
    '''
    construct a graph or nice format from test_2 
    '''

if __name__ == '__main__':
 
    df23384 = df_test[df_test['City_Type']==23384]
    df1216 = df_test[df_test['City_Type']==1216]
    df1036 = df_test[df_test['City_Type']==1036]
    df1704 = df_test[df_test['City_Type']==1704]
    df2662 = df_test[df_test['City_Type']==2662]
    df1729 = df_test[df_test['City_Type']==1729]
    df1217 = df_test[df_test['City_Type']==1217]
    df1352 = df_test[df_test['City_Type']==1352] 
    df2517 = df_test[df_test['City_Type']==2517]
    df816 = df_test[df_test['City_Type']==816] 

    df_one = df_test[ (df_test['Second']== 0) & (df_test['Third']==0) ]
    df_two = df_test[df_test['Second']== 1]
    df_three = df_test[df_test['Third']== 1]

    event_list = [df_one,df_two,df_three]
    city_list = [df23384,df1216,df1036,df1704,df2662,df1729,df1217,df1352,df2517,df816 ]
    # for i in city_list:
    #     
    #     get = run_test_typeB(i)
          #get2 = run_test_typeC(i)
    #     print(f'this is the output {get2}', len(i))

    for i in event_list:
        #get_it = run_test_typeB(i)
        get_it2 = run_test_typeC(i)
        print(f'this is the output {get_it2}', len(i))
     


    
    #print(run_test_typeA(X_train,y_train)) 
    #print(run_test_typeB(X_train,y_train))
    #print(run_test_typeD(X_train,y_train))

    
    # ok = run_test_typeC(X_train,y_train)
    #print(ok)
    # lst = [] 
    # for i in ok: # for item in printout
    #     for ii in i:
    #         lst.append(ii)
    #print(lst)

            

     
    
