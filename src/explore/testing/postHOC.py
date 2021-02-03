import pandas as pd 
import numpy as np 
pd.set_option('display.max_columns', None) 
df = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/ready12_24_train.csv')  
from preprocessing import drop_cols , one_hot_encoding , scale
from city_testing import run_test_typeAA , run_test_typeC
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


def run_tests(dataframe):
    '''
    get df, make copy, run tests, store results
    '''
    dataframe['Patient ID'] = ids['Patient_ID'].values
    print(dataframe.head(10))
    get =  run_test_typeAA(dataframe)
   





    return get

if __name__ =='__main__':
    output = run_tests(df_no_scale)
    print(output)


'''
As of 1/31/21 City_test AA is being edited: prediction,proba, ytarget are put back on test df
--need to edit / mess with patient IDs within that function 


'''
 

