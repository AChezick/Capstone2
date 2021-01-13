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
import matplotlib.pyplot as plt
#from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV
 
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/ready12_24_train.csv')  
df["Camp Start Date - Registration Date"] = df['delta_first_reg']
df[ "Registration Date - First Interaction"] = df['interaction_regreister_delta']
df["Camp Start Date - First Interaction"]=df['delta_first_start']
df["Camp End Date - Registration Date"]=df['delta_reg_end']
df['Camp Length'] = df['Camp_Length']
rename=['delta_first_reg','delta_reg_end','Camp_Length','interaction_regreister_delta','delta_first_start', 'Camp_length']
df =df.drop(rename,axis=1)

from preprocessing import drop_cols , one_hot_encoding , scale , drop_cols_specific
data = drop_cols(df) # drop cols
hot_code = data.copy() 
data2 = drop_cols_specific(df) #drop cols for camp testing
dataframe_1 = scale(data) #Scale all features that are not binary 
dataframe_2 = scale(data2) 

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



def run_test_typeC(dataframe): # might need to rescale / hotencode / drop those
    '''
    do part 2 _ maybe optomize  ? 
    '''
    y = dataframe.pop('y_target')
    X = dataframe 
    camp_ID = X.Health_Camp_ID.max()  
    del X['Health_Camp_ID'] 
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

    
    mat1 = confusion_matrix(y_test,log_thresh1 ) 
    mat2 = confusion_matrix(y_test,log_thresh2 ) 
    mat3 = confusion_matrix(y_test,log_thresh3 ) 

    y_counts = sum([1 for x in y_test.values if x ==1])

    conf_matrix1,conf_matrix2,conf_matrix3 = [],[],[]

  
    
    test_size = len(X_test)
    return [str(camp_ID), mat1[0][1],mat1[1][1], y_counts , test_size ] #array of ratio, total_ytarget, total size , camp length , 

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
    hot_code2 = one_hot_encoding(hot_code, columns = ['Category1_x','Category2','Category3','Job_Type', 'online_score', 'City_Type'])
    hot_code2.to_csv('hot_code_NO_scale.csv',index=False)
    print(hot_code2)
    # df_encode= one_hot_encoding(dataframe_1, columns = ['Category1_x','Category2','Category3','Job_Type', 'online_score']) # for testing differnt cities dont drop City_Type
    # df_encode2= one_hot_encoding(dataframe_2, columns = ['Category1_x','Category2','Category3','Job_Type', 'online_score'])

    # df_encode1 = df_encode.copy() 
    # df_encode4 = df_encode2.copy() 

    # df_test = df_encode1.drop(['Category1_x','Category2','Category3','Job_Type', 'online_score'],axis=1)
    # df_test1 = df_test.copy()  

    # df_test2 = df_encode4.drop(['Category1_x','Category2','Category3','Job_Type', 'online_score'],axis=1)
    # df_test4 = df_test2.copy() 
    
 
    # df23384 = df_test[df_test['City_Type']==23384]
    # df1216 = df_test[df_test['City_Type']==1216]
    # df1036 = df_test[df_test['City_Type']==1036]
    # df1704 = df_test[df_test['City_Type']==1704]
    # df2662 = df_test[df_test['City_Type']==2662]
    # df1729 = df_test[df_test['City_Type']==1729]
    # df1217 = df_test[df_test['City_Type']==1217]
    # df1352 = df_test[df_test['City_Type']==1352] 
    # df2517 = df_test[df_test['City_Type']==2517]
    # df816 = df_test[df_test['City_Type']==816] 

    # df_one = df_test[ (df_test['Second']== 0) & (df_test['Third']==0) ]
    # df_two = df_test[df_test['Second']== 1]
    # df_three = df_test[df_test['Third']== 1]

    # event_list = [df_one,df_two,df_three]
    # city_list = [(df23384,23384),(df1216,1216),(df1036,1036),(df1704,1704),(df2662,2662),
    # (df1729,1729),(df1217,1217),(df1352,1352),(df2517,2517),(df816,816) ]
    # camp_list = [6578, 6532, 6543, 6580, 6570, 6542, 6571, 6527, 6526, 6539, 6528,
    #     6555, 6541, 6523, 6538, 6549, 6586, 6554, 6529, 6540, 6534, 6535, 6561, 6585, 
    #     6536, 6562, 6537, 6581, 6524, 6587, 6557, 6546, 6569, 6564, 6575, 6552, 6558, 
    #     6530, 6560, 6531, 6544, 6565, 6553, 6563]
    # resultz_dict = {} 
    # for i in camp_list:
    #     get = df_test4[df_test4['Health_Camp_ID'] == i] #make data_frame for analysis 
    #     get_y_counts = get[get['y_target']==1] # get count for total number of attends 
    #     size = len(get)  # obtain size of data_frame 
    #     test = run_test_typeC(get) # create object that has results from data_frame testing
    #     i_ = str(i)
    #     resultz_dict[i_] = test
    # print(resultz_dict)
    
    # camp_df = pd.DataFrame.from_dict(resultz_dict, orient='index', columns = ['Health_Camp_ID',
    #     'False Positive', 'False Negative', 'Y Count', 'Test Size'])
    # camp_df.to_csv('results_by_Camp.csv',index=False)



    # result_by_city_dict = {}
    # for i in city_list:
    #     get = i[0] #make data_frame for analysis 
    #     print(get.City_Type.unique)
    #     get_y_counts = get[get['y_target']==1] # get count for total number of attends 
    #     size = len(get)  # obtain size of data_frame 
    #     test = run_test_typeC(get) # create object that has results from data_frame testing
    #     i_ = str(i_[0])
    #     result_by_city_dict[i_] = test
    # print(result_by_city_dict)
     

    # city_list = pd.DataFrame.from_dict(result_by_city_dict, orient='index', columns = ['Health_Camp_ID',
    # 'False Positive', 'False Negative', 'Y Count', 'Test Size'])
    # city_list.to_csv('results_by_CITY.csv',index=False)

    # result_by_TYPE_dict = {}
    # for i in event_list:
    #     get = df_test4[df_test4['Health_Camp_ID'] == i] #make data_frame for analysis 
    #     get_y_counts = get[get['y_target']==1] # get count for total number of attends 
    #     size = len(get)  # obtain size of data_frame 
    #     test = run_test_typeC(get) # create object that has results from data_frame testing
    #     i_ = str(i)
    #     result_by_city_dict[i_] = test
    # print(result_by_city_dict)
     

    # city_list = pd.DataFrame.from_dict(result_by_city_dict, orient='index', columns = [ 'Health_Camp_ID',
    # 'False Positive', 'False Negative', 'Y Count', 'Test Size'])
    # city_list.to_csv('results_by_CITY')

     
    # for i in event_list:
    #     #get_it = run_test_typeB(i)
    #     get_it2 = run_test_typeC(i)
    #     print(f'this is the output {get_it2}', len(i))

        
    

    

      
    #     conf_matrix1,conf_matrix2,conf_matrix3 = [],[],[]
    #     for matrix in test: # for item in printout
    #         for i in matrix: # for array 1 , for array2
    #             for ii in i:
    #                 conf_matrix1.append(ii) #list. append num
    #         results_dict[i]=conf_matrix1
    # print(results_dict)
    # 
