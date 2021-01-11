import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler =  StandardScaler()
pd.set_option('display.max_columns', None) 
#df = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/ready12_24_train.csv') 
def drop_cols(df):
    '''
    Drop columns 
    '''

    drop_thez=[ 'Patient_ID_x', 'Health_Camp_ID', 'Registration_Date', 'Category1_y','Camp_Start_Date2', 
    'Camp_End_Date2', 'patient_event', 'Unnamed: 0_x','Unnamed: 0.1_x', 'Online_Follower_x', 'First_Interaction'
    ,'Employer_Category' , 'Event1_or_2_x' ,'Category1_y', 'Unnamed: 0_y', 'Unnamed: 0.1_y', 'Patient_ID_y',
    'Online_Follower_y', 'Event1_or_2_y','Health Score', 'Camp_length',
    'Number_of_stall_visited', 'Last_Stall_Visited_Number']

    df_ = df.drop(drop_thez, axis =1)
    return df_

#Why_Not_Do_This < because the word 'day' was still rpesent in datetime object?
def scale(df): 
    '''
    Scale columns that are non-ordinal 
    '''
    columnz = ['Var1','Var2', 'Var3', 'Var4', 'Var5', 'Camp_Length', 'delta_reg_end',
    'delta_first_reg','interaction_regreister_delta', 'delta_first_start']

    for i in columnz:
        i_ = df[i].to_frame() 
        
        i_i = scaler.fit_transform(i_)
        df[i] = i_i
    return df  

def one_hot_encoding(df, columns):
    '''
    Hot encoding of categorical columns 
    '''
    hot_df = df.copy()
    for i in columns:
        dummies = pd.get_dummies(df[i], drop_first=True)
        hot_df = pd.concat([hot_df, dummies[:]], axis=1)

    #hot_df = hot_df.drop(columns,axis=1)
    return hot_df 

if __name__ =="__main__":
    dropped = drop_cols(df) 
    scaled_df = scale(dropped)
    #df_encode = one_hot_encoding(scaled_df, columns = ['City_Type','Category1_x','online_score','Category2','Category3', 'Job_Type'])
    print(df_encode)
 