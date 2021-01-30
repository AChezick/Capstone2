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

df = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/ready12_24_train.csv')  
# df["Camp Start Date - Registration Date"] = df['delta_first_reg']
# df[ "Registration Date - First Interaction"] = df['interaction_regreister_delta']
# df["Camp Start Date - First Interaction"]=df['delta_first_start']
# df["Camp End Date - Registration Date"]=df['delta_reg_end']
# df['Camp Length'] = df['Camp_Length']
# rename=['delta_first_reg','delta_reg_end','Camp_Length','interaction_regreister_delta','delta_first_start', 'Camp_length']
# df =df.drop(rename,axis=1)

from preprocessing import drop_cols , one_hot_encoding , scale
#print(df)
#data = drop_cols(df) # drop cols
dataframe_1 = scale(df) #Scale all features that are not binary 

import matplotlib.pyplot as plt

def create_holdout(dataframe ):
    '''
    Create holdout dataframe 
    '''
    dataframe.y_target = dataframe.y_target.astype(int)
    y = dataframe.pop('y_target')
    #del dataframe['y_target']
    X = dataframe 
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, 
                                                        shuffle=False,
                                                        test_size=0.3, 
                                                        random_state=101)

    return X_train, X_holdout, y_train, y_holdout

def run_test_typeA(dataframe): # this test might have to be done differently than the other two for the holdout aspect 
    '''
    Remove columns, print list of columns, drop columns, run model
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
     
    preds_xg1_thresh1, preds2_xg1_thresh2 = xg_reg1_proba>=0.35 , xg_reg1_proba>=0.45


    # xg_reg1_disp = plot_roc_curve(xg_reg1 , X_testD, y_testD)
    # plt.show()

    print(classification_report(y_testD,xg_reg1_predict ) )
    print(confusion_matrix(y_testD,preds_xg1_thresh1))  

    print(confusion_matrix(y_testD,preds2_xg1_thresh2))  



def run_test_typeK(dataframe): # this test might have to be done differently than the other two for the holdout aspect 
    '''
    Remove columns, print list of columns, drop columns, run model
    '''
    y = dataframe.pop('y_target')
    x = dataframe
    del x['Health_Camp_ID']
    data_dmatrix = xgb.DMatrix(data=x,label=y) 
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=101) 
    
    svc = SVC(random_state=42)
    svc.fit(X_test, y_test)
    svc_preds = svc.predict(X_test)

    print(classification_report(y_test,svc_preds) )
    svc_disp = plot_roc_curve(svc, X_test, y_test)
    plt.show()

def run_test_typeB(dataframe):
    '''
    run the test
    '''
    y = dataframe.pop('y_target')
    x = dataframe
    del x['Health_Camp_ID']
    data_dmatrix = xgb.DMatrix(data=x,label=y) 
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=101) 

    # xg_reg1 = XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
    #             max_depth = 6, alpha = 6, n_estimators = 12, eval_metric = 'auc')

    # xg_reg1.fit(X_train,y_train) 
    # preds = xg_reg1.predict(X_test) 
    # print(classification_report(y_test,preds))

    params = {"objective":'binary:logistic','colsample_bytree': 0.6,'learning_rate': 0.1,
       "min_child_weight": 5 ,'max_depth': 6, 'alpha': 10, 'eval_metric':'auc', 'subsample':0.8} 
    cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=5,
                    num_boost_round=100,early_stopping_rounds=100, metrics='auc', as_pandas=True, seed=123)
    xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)
    print(cv_results[95:])
    xgb.plot_importance(xg_reg, title = None ,importance_type="weight", grid=False,max_num_features=6,xlabel='Feature Score', ylabel= None, height=.7)
    
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.rcParams['figure.dpi'] = 300 
    
    plt.show()

    # xgb.fit(X_test,y_test) 
    # preds = xgb.predict(X_test) 
    # print(classification_report(y_test,preds))

    return None 




def run_test_typeC(dataframe):
    '''
    do part 2 _ maybe optomize  ? 
    '''
    y = dataframe.pop('y_target')
    X = dataframe 
    X_train, X_test, y_train,y_test = train_test_split(
    X, y , test_size = 0.5, random_state=101)

    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)
    LogisticRegression(penalty='l1', dual=False, C=1.0, 
        fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, 
        solver='saga', max_iter=None , multi_class='auto', verbose=0, warm_start=False, 
        n_jobs=-1, l1_ratio=None) 
    
    predictions = logmodel.predict(X_test)
    print(classification_report(y_test,predictions))
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

    df_encode = one_hot_encoding(dataframe_1, columns = ['City_Type2_x','Category 1','Category 2','Category 3','Job Type_x', 'online_score'])
    df_encode1 = df_encode.copy() 

    df_test = df_encode1.drop(['City_Type2_x','Category 1','Category 2','Category 3','Job Type_x', 'online_score'],axis=1)
    #df_test3.to_csv(('/home/allen/Galva/capstones/capstone2/src/explore/temp_csv/ready_to_test5.csv'))
    
    df_test3 = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/temp_csv/ready_to_test5.csv')
    print(df_test3.columns)
    df_test3 = df_test3.drop(['Camp Length', 'Second', 'Third',
     'C', 'D', 'E', 'F', 'G', '2100', 'BFSI', 'Broadcasting',
       'Consulting', 'Education', 'Food', 'Health', 'Manufacturing', 'Others',
       'Real Estate', 'Retail', 'Software Industry', 'Technology', 'Telecom'],axis=1 ) 
 
    X_train, X_holdout, y_train, y_holdout= create_holdout(df_test) 
    #print(X_train, X_holdout, y_train, y_holdout)
    df_test1, df_test2 = df_test3.copy() , df_test3.copy()
    # #df_test1a = combine_info( df_test,dropped=['Var4','7.0','4','2100','2517','1704','1216'])
    print(run_test_typeA(dataframe=df_test3))
    print(run_test_typeB(dataframe= df_test3))
    print(run_test_typeC(dataframe = df_test3))
            
     
    
    
   