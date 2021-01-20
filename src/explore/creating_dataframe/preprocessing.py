import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler =  StandardScaler()
pd.set_option('display.max_columns', None) 
df1 = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/ready12_24_train.csv') 
df=df1.copy() 
def drop_cols(df):
    '''
    Drop columns 
    '''
    # Health_Camp_ID 
    drop_thez=[ ]

    df_ = df.drop(drop_thez, axis =1)
    return df_

def drop_cols_specific(df):
    '''
    Drop columns: edit for dropping features and testing for improvement - personal grid searching 
    '''
    #Health_Camp_ID 
    drop_thez=[ 'Patient_ID_x', 'Registration_Date', 'Category1_y','Camp_Start_Date2', 'Camp_length',
    'Camp_End_Date2', 'patient_event',  'Online_Follower_x', 'First_Interaction'
    ,'Employer_Category' ,'Category1_y',  'Patient_ID_y', 'Category2_y', 'Category3_y',
    'Online_Follower_y' ,'Health Score', 'City_Type2_y', 'City_Type',
    'Number_of_stall_visited', 'Last_Stall_Visited_Number','Patient_ID_y']

    df_ = df.drop(drop_thez, axis =1)
    return df_

#Why_Not_Do_This < because the word 'day' was still rpesent in datetime object?
def scale(df): 
    '''
    Scale columns that are non-ordinal 
    '''
    columnz = ['Var1','Var2', 'Var3', 'Var4', 'Var5', 
       'Camp Start Date - Registration Date',
       'Registration Date - First Interaction',
       'Camp Start Date - First Interaction',
       'Camp End Date - Registration Date', 'Camp Length']

    for i in columnz:
        i_ = df[i].to_frame() 
        
        i_i = scaler.fit_transform(i_)
        df[i] = i_i
    return df  

def one_hot_encoding(df, columns ): #=[ 'online_score','Category 1','Category 2', 'Category 3']
    '''
    Hot encoding of categorical columns 
    '''
    hot_df = df.copy()
    for i in columns:
        if i != 'Category 2':
            dummies = pd.get_dummies(df[i], drop_first=True)
            hot_df = pd.concat([hot_df, dummies[:]], axis=1)
        else:
            dummies = pd.get_dummies(df[i], drop_first=False)
            hot_df = pd.concat([hot_df, dummies[:]], axis=1)

    del hot_df['B']
    # columns2 =['Job Type_x']

    # for i in columns2:
    #     dummies = pd.get_dummies(df[i], drop_first=False)
    #     hot_df = pd.concat([hot_df, dummies[:]], axis=1)

    # del hot_df['Real Estate']
     
    return hot_df 

if __name__ =="__main__":
    df2=one_hot_encoding(df1)
    print(df2)
  