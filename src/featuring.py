import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, plot_roc_curve, accuracy_score

pd.set_option('display.max_columns', None) 
patient_df = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Patient_Profile.csv')
patient_df1 = patient_df.copy()

'''
# Lines 33 and 34 IN PYTHON 3.8 will work for replacing Nones # 
# 33 - 46 Are for exploring numbers of people who are in categories 
c1 = patient_df1[ (patient_df1['Age'] == 999)    &   (patient_df1['LinkedIn_Shared'] ==1)   ]
c1v=c1['Age'].values
c21 = len(c1v)
c1vv = [int(i) for i in c1v]
avg_ = sum(c1vv) / c21
'''
patient_df1['Age'].replace(to_replace="None", value=np.nan, inplace=True)
patient_df1['Age'].fillna(value = 0, inplace=True) 
patient_df1['Age'] = patient_df1['Age'].map(lambda x: 1 if x != 0 else 0)
#print(patient_df1.head(10))
lsts = ['Age' , 'Education_Score', 'City_Type', 'Income']

def make_edits(lsts, patient_df1):
    for i in lsts:
        patient_df1[i].replace(to_replace="None", value=np.nan, inplace=True)
        patient_df1[i].fillna(value = 0, inplace=True) 
        if i != 'Income':
            patient_df1[i] = patient_df1[i].map(lambda x: 1 if x != 0 else 0)
        else:
            break
    return patient_df1

patient_df2 = make_edits(lsts, patient_df1)

