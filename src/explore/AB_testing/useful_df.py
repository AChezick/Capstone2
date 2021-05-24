import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
from preprocessing_formatting import drop_cols , one_hot_encoding , scale
 


pd.set_option('display.max_columns', None) 

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

def make_useful_df():
    

    return dataframe

print(dataframe.head(), dataframe.info() , dataframe.describe( ))
'''
from preprocessing_formatting import drop_cols , one_hot_encoding , scale
#from city_testing import run_test_typeC
import itertools as it 

hot_scale = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/temp_csv/train_4_model.csv')
df_no_scale = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/temp_csv/hot_code_NO_scale.csv')

'''
# What I want is the 75244, hot encoded, all date features removed , patient_IDS, Camp_IDs , scaled 