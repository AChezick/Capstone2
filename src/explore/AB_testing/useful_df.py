import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
from preprocessing_formatting import drop_cols , one_hot_encoding , scale
 


pd.set_option('display.max_columns', None) 
dff = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/model_agent.csv')
df_withdates = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/placeholder/df_withdates.csv') 
df_blank = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/ready12_24_train.csv') 
# Dataframe = # Find a csv with scaled/ data ready 
ids =  pd.read_csv('/home/allen/Galva/capstones/capstone2/data/ready12_24_train.csv')
#del df_no_scale['Health_Camp_ID']
dataframe = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/temp_csv/ab_df.csv')  
 
start = df_withdates['Camp_Start_Date2'].values
end = df_withdates['Camp_End_Date2'].values

def edit_df():
    '''
    create new cols 
    '''
    print(df_withdates.columns)
    data = drop_cols(df_blank) # drop date cols 
    df_encode = one_hot_encoding(data, columns = ['City_Type2_x','Job Type_x','Category 2','Category 3','Category 1', 'online_score'])
    df_encode['Start'] = start
    df_encode['End'] = end
    df_encode['Patient_ID'] = df_withdates['Patient_ID_x'].values
    df_encode.to_csv('/home/allen/Galva/capstones/capstone2/src/explore/temp_csv/ab_df.csv')
    return df_encode 



def happen():
    df = dff.copy()
    lst = [ 'prediction_kNN', 'prediction', 'prediction_sVC', 'prediction_xg' ]

    for col in lst:
        col_ = col + '_score'
        print(col_)
        df['and_'+col] = dff['y_target'] + dff[col]
    for col in lst:
        df['and_'+col] = df['and_'+col].apply(lambda x: 2 if x ==0 else x) 
    for col in lst:
        col_ = col + '_score'
        df[col_] = df['and_'+col].apply(lambda x: 1 if x == 2  else 0  )
    
    df['Score'] = df['prediction_kNN_score']+ df['prediction_score'] +df['prediction_sVC_score'] + df['prediction_xg_score'] 
    
    return df





# print(dataframe.head(), dataframe.info() , dataframe.describe( ))
if __name__ =='__main__':
    check = happen()
    print(check.head(20))


'''
from preprocessing_formatting import drop_cols , one_hot_encoding , scale
#from city_testing import run_test_typeC
import itertools as it 

hot_scale = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/temp_csv/train_4_model.csv')
df_no_scale = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/temp_csv/hot_code_NO_scale.csv')

'''
# What I want is the 75244, hot encoded, all date features removed , patient_IDS, Camp_IDs , scaled 