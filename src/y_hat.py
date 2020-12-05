
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, plot_roc_curve, accuracy_score
import itertools as it 
from itertools import combinations 
pd.set_option('display.max_columns', None) 

pd.set_option('display.max_columns', None) 

class Get_make_dataFrame():
    """Consolidate data & make target column."""

    def __init__(self):
        self
'''
sd={}
for i in patient_df['Employer_Category'].values:
    if i not in sd:
        sd[i]=1
    else:
        sd[i]+=1


patient_df1 = patient_df.copy()

m = {'Software Industry': 1, 'BFSI':2 , 'Education':3 , 'Others':4 , 'Technology': 5, 'Consulting':6 , 'Manufacturing':7, 'Health':8, 'Retail':9,  'Transport': 10 , 'Broadcasting': 11, 'Food':12 , 'Telecom': 13, 'Real Estate':14}
patient_df1['Job_Type'] = patient_df1['Employer_Category'].map(m) 
patient_df1['Job_Type'].fillna(value = 9999, inplace=True)

def impute(x):
    for _ in x:
        if _ in [1,2,3,4,5,6,7,8,9]:
            return 1
        else:
            return 0

patient_df1['Education_Score'].fillna(value=0, inplace=True) 

patient_df1['Income'].fillna(value=0, inplace=True)
'''


if __name__ == '__main__' : 


event1 = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/First_Health_Camp_Attended.csv')
event2 = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Second_Health_Camp_Attended.csv')
event3 = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Third_Health_Camp_Attended.csv')
patient_df = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Patient_Profile.csv')



