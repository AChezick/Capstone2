'''
This file combines the patient data frame and tran data frame
to create the master train data frame

Features are scaled 

'''
import numpy as np 
import pandas as pd 
pd.set_option('display.max_columns', None) 
df =pd.read_csv('/home/allen/Galva/capstones/capstone2/data/D7.csv')

df['Edu1'] = df['Education_Scorez'].apply(lambda x:1 if x == 1 else 0)
df['Edu2'] = df['Education_Scorez'].apply(lambda x:1 if x == 2 else 0)

train = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Train.csv')
train1 = train.copy()
test_df = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/test.csv') 
test_df1 = test_df.copy() 

train1_ = ['Var1','Var2','Var3','Var4','Var5'] 
def make_frames(x, train1_):

    for i in train1_:
        x[i]

    return x
test_frames = make_frames(test_df1, train1_)
train_frames = make_frames(train1, train1_ )

from sklearn import preprocessing
scaler = StandardScaler() 

# class PreProcessing(): 

#     def __init__(self,)

# def scaling(train1,test_df1, train1_):
#     for i in train1_:
#         transform_vals = [] 
#         m =  sum(train[i].values) / len(train1[i].values)
#         s = train1_[i].values
#         sd = np.std(s)

#         for ii in i:
#             ii_ = ii-m / sd
#             transform_vals.append(ii_)
#         train1[i] = transform_vals 
#         test_df1[i] = transform_vals 
#     return test_df1 ,  train1

#     if __name__ == '__main__


# print(scaling(train1,test_df1,train1_)) 
# print(test_df1.head(), train1.head() )