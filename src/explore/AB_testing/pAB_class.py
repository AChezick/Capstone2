
# 10/20/21
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
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error

from preprocessing_formatting import drop_cols , drop_cols_specific, one_hot_encoding , scale , scale2    
#from city_testing import run_test_typeC
import itertools as it 

class ABmodeling:

    def __init__(self): #df_passed1
        # self.df1 = df_passed1 [] How to initlize a passed data frame to this class <?> 
        self.df = pd.read_csv('/home/allen/RIP_Tensor1/capstone2/Capstone2/src/explore/ab_df.csv')
        self.df1 = pd.read_csv('/home/allen/RIP_Tensor1/capstone2/Capstone2/src/explore/ab_df.csv')

        # self.df_encode1 = self.df.drop(['City_Type2_x','Job Type_x','Category 2','Category 3','Category 1', 'online_score', 'Start','End'],axis=1) 
        # self.df1,self.df2 = self.df_encode1.copy() , self.df_encode1.copy() 
        # self.ans ={}
        self.df_encode1 = self.df1.drop(['City_Type2_x','Job Type_x','Category 2','Category 3','Category 1', 'online_score', 'Start','End'],axis=1) 
        self.df1,self.df2 = self.df_encode1.copy() , self.df_encode1.copy() 
        #self.test_df = self.df1[self.df1['Health_Camp_ID'] == 6585 ]
        for index,item in enumerate([(6544, ['NA', 6530, 6560]) , (6561, ['NA', 6530, 6560, 6544]), (6585, ['NA', 6530, 6560, 6544])]): 
            if len(item[1]) <=1: #
                break
        else:
            iD = item[0] # Camp_ID
            camps = item[1][1:] #list of camps for training 

            self.test_df1 = self.df1[self.df1['Health_Camp_ID'] == iD ]  
            self.train_df = self.df2.loc[self.df2['Health_Camp_ID'].isin(camps)  ]  
            
        self.test_dfl , self.train_dfl = self.test_df1.copy() , self.train_df.copy()
        self.test_dfk , self.train_dfk = self.test_df1.copy() , self.train_df.copy()


    def run_tests(self ):
        '''
        get dfs, make copies, run tests , combine_results, send back for AB testing
        '''
        self.final_df = self.test_df.copy() 
        print(self.final_df.columns , 'these are the columns currently being modeled other than  [Health_Camp_ID]/Patient_ID] ')
        self.test_df1 , self.train_df1 = self.test_df.copy() , self.train_df.copy() 
  
        get_L =  beta.run_test_typeL()
        get_k = beta.run_test_typek() 

        self.final_df['log'] = get_L['predictionL']
        self.final_df['log_proba'] = get_L['probaL']
        self.final_df['kmean'] = get_k['predictionK']
        self.final_df['k_proba'] = get_k['probaK']

        return self.final_df

    def run_test_typeL(self):
        '''
        run knn for post hoc - analysis 
        '''
        self.train_y = self.train_dfl['y_target'].values #getting y_target values for test & train
        # test_y = test_dfl['y_target'].values 

        del self.train_dfl['y_target'] #deleting values 
        del self.test_dfl['y_target']

        df_train = self.train_dfl
        df_test = self.test_dfl
        
        del df_train['Health_Camp_ID'] 
        del df_test['Health_Camp_ID']  

        del df_train['Patient_ID'] 
        del df_test['Patient_ID'] 
        
        w = {0:65, 1:35} 
        logmodelx = LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=1.0, 
                fit_intercept=True, intercept_scaling=1, class_weight=w , random_state=None, 
                solver='lbfgs', max_iter=50, multi_class='auto', verbose=0, warm_start=False, 
                n_jobs=-1, l1_ratio=None ) 
        logmodelx.fit(df_train, self.train_y)

        pure_probaz = logmodelx.predict_proba(df_test) 
        predictionsz = logmodelx.predict(df_test)
        preds2x = predictionsz >= .3


        df_test['predictionL'] = predictionsz   
        df_test['probaL'] =  pure_probaz[:,-1]     
        #df_test['predictionL2']  = preds2x  
        return df_test 

    def run_test_typek(self):
        '''
        run knn for post hoc - analysis 
        '''
        traink_y = self.train_dfk['y_target'].values #getting y_target values for test & train
        testk_y = self.test_dfk['y_target'].values 

        del self.train_dfk['y_target'] #deleting values 
        del self.test_dfk['y_target']

        traink_y = traink_y 
        testk_y = testk_y

        df_train = self.train_dfk
        df_test = self.test_dfk
        
        #might need to delete / edit DF
        del df_train['Health_Camp_ID'] 
        del df_test['Health_Camp_ID']  

        del df_train['Patient_ID'] 
        del df_test['Patient_ID'] 

        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(df_train,traink_y)
        knn_preds = knn.predict(df_test)
        knn_proba = knn.predict_proba(df_test) 

        df_test['predictionK'] = knn_preds
        df_test['probaK'] = knn_proba[ :,1]   
        #df_test['y_target'] = test_y

        return df_test 




if __name__ =='__main__':

    beta = ABmodeling( )
    checkl = beta.run_test_typeL()
    checkk = beta.run_test_typek()
    check2 = beta.run_tests()
    print(check2)


'''
if I pass a list of data frames into the class
|||For each df|||
- make 4 copies
- every copy is sent to one model
- the model sends back the copied df
- results from the 4 tests are added back to OG

10-23-21
Seems like I will need to re plan how I initalize the objects and pass them ,
each declared global within __init__ but the local functions are not able find them
-successful beta run for logistic regression 
--Something with the global values and the y_target needing to be deleted for each model
  missing two columns , i think that some errors are coming from variable naming

10-24-21
--did some renaming ,there is an error, just figure out how to pass
'''