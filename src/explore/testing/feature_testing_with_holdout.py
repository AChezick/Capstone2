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
# data = drop_cols(df) # drop cols
# dataframe_1 = scale(data) #Scale all features that are not binary 

import matplotlib.pyplot as plt

df_no_scale = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/hot_code_NO_scale.csv')

del df_no_scale['Health_Camp_ID']
del df_no_scale['Category1_x']
del df_no_scale['Category2']

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

def run_test_typeB(X_testz,y_trainz): # THIS NEEDS TO be split for train and test somehow .
    '''
    run the test
    '''
    y = y_trainz
    x = X_testz
    data_dmatrix = xgb.DMatrix(data=x ,label=y )  
    X_trainD, X_testD, y_trainD, y_testD = train_test_split(x, y, test_size=0.2, random_state=101)  

    params = {'objective':'binary:logistic','colsample_bytree': 0.6,'learning_rate': 0.1,
       "min_child_weight": 5 ,'max_depth': 6, 'alpha': 10, 'eval_metric':'auc', 'subsample':0.8} 
    cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=5,
                    num_boost_round=50,early_stopping_rounds=100, metrics='auc', as_pandas=True, seed=101)
    xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

    #print(cv_results[45:])
    xgb.plot_importance(xg_reg )
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.rcParams['figure.dpi'] = 200 
    plt.show()

    # xg_reg.fit(X_trainD,y_trainD, early_stopping_rounds=10, eval_metric='auc', eval_set=[X_testD,y_testD])
    # xg_preds = xg_reg.predict(X_testD)
    # xg_probaz = xg_reg.predict_proba(X_testD)[:,1] 

    # predsD , preds2D = xg_probaz  >= .5 , xg_probaz   >= .4

    # print(classification_report(y_testD,xg_preds) )
    # print(classification_report(y_testD,preds2D ))
    
    # xgd_disp = plot_roc_curve(xg_reg, X_testD, y_testD)
    # plt.show()

    
    return cv_results

def run_test_typeD(dataframe):
    '''
    run an XDG Classifier  
    '''
    
    y = dataframe.pop('y_target')
    x = dataframe

    X_trainD, X_testD, y_trainD, y_testD = train_test_split(x, y, test_size=0.3, random_state=101) 
     
    data_dmatrix_train = xgb.DMatrix(data=X_trainD,label=y_trainD) # not sure what these are doing ?! 
    data_dmatrix_test = xgb.DMatrix(data=X_testD,label=y_testD) 

    xg_reg1 = XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 6, alpha = 8, n_estimators = 12, eval_metric = 'auc', label_encoder=False)

    xg_reg1.fit(X_trainD,y_trainD) 

    xg_reg1_predict = xg_reg1.predict(X_testD) #(0/1 associated with .5)
    xg_reg1_proba = xg_reg1.predict_proba(X_testD)[:,1]
    #print(xg_reg1_proba, 'this is the predict proba')
    preds_xg1_thresh1 =xg_reg1_proba>=0.5
    preds2_xg1_thresh2 = xg_reg1_proba>=0.35



    xg_reg1_disp = plot_roc_curve(xg_reg1 , X_testD, y_testD)
    plt.show()
 
    

    print(classification_report(y_testD,preds_xg1_thresh1 ) )
    print(confusion_matrix(y_testD,preds_xg1_thresh1))  
    print(confusion_matrix(y_testD,preds2_xg1_thresh2))  

    return None 


def run_test_typeC(X_testz,y_trainz):
    '''
    do part 2 _ maybe optomize  ? 
    '''
    y = y_trainz
    X = X_testz 
    X_trainc, X_testc, y_trainc,y_testc = train_test_split(
    X, y , test_size = 0.2, random_state=101)

    logmodel = LogisticRegression()
    logmodel.fit(X_trainc,y_trainc)
    LogisticRegression(penalty='l1', dual=False, C=1.0, 
        fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, 
        solver='saga', max_iter=None , multi_class='auto', verbose=0, warm_start=False, 
        n_jobs=-1, l1_ratio=None) 
    
    predictions = logmodel.predict(X_testc)
    confs_output = confusion_matrix(y_testc,predictions) 
    return confs_output  

if __name__ == '__main__':

    df_encode = one_hot_encoding(dataframe_1, columns = ['City_Type','Category1_x','Category2','Category3','Job_Type', 'online_score'])
    df_encode1 = df_encode.copy() 

    df_test = df_encode1.drop(['City_Type','Category1_x','Category2','Category3','Job_Type', 'online_score'],axis=1)
     
    df_test1, df_test2 = df_test.copy() , df_test.copy()

    X_train, X_holdout, y_train, y_holdout= create_holdout(df_no_scale) 
 
    print(run_test_typeA(X_train,y_train))
    print(run_test_typeB(X_train,y_train))
    print(run_test_typeD(df_no_scale))

            

     
    
    # THESE ARE WORKING #
    # df_test.to_csv('train_4_model.csv', index = False)
     
