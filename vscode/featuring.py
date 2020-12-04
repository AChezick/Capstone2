import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, plot_roc_curve, accuracy_score

pd.set_option('display.max_columns', None) 

event1 = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/First_Health_Camp_Attended.csv')
event2 = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Second_Health_Camp_Attended.csv')
event3 = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Third_Health_Camp_Attended.csv')
patient_df = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Patient_Profile.csv')
patient_df1 = patient_df.copy()

# event1_ids, event2_ids = event1['Patient_ID'].values , event2["Patient_ID"].values
# event_1_and_2_ids = []
# for i in event1_ids:
#     event_1_and_2_ids.append(i)
# for i in event2_ids:
#     event_1_and_2_ids.append(i)

ugh = event3[ (event3['Number_of_stall_visited'] == 0) & (event3['Patient_ID'] != 0 ) ]
print(ugh['Patient_ID'].values ) 

#patient_df1['Event1_or_2'] = patient_df1['Patient_ID'].apply(lambda x:1 if x in event_1_and_2_ids else 0 )
#patient_df1['City_Type'].fillna(value = 'Z', inplace=True)
m = {'Software Industry': 1, 'BFSI':2 , 'Education':3 , 'Others':4 , 'Technology': 5, 'Consulting':6 , 'Manufacturing':7, 'Health':8, 'Retail':9,  'Transport': 10 , 'Broadcasting': 11, 'Food':12 , 'Telecom': 13, 'Real Estate':14}
patient_df1['Job_Type'] = patient_df1['Employer_Category'].map(m) 
patient_df1['Job_Type'].fillna(value = 9999, inplace=True)
patient_df1['Education_Score'].fillna(value=0, inplace=True) 
patient_df1['Income'].fillna(value=0, inplace=True)

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

