import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, plot_roc_curve, accuracy_score
import itertools as it 
from itertools import combinations 
pd.set_option('display.max_columns', None) 

event1 = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/First_Health_Camp_Attended.csv')
event2 = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Second_Health_Camp_Attended.csv')
event3 = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Third_Health_Camp_Attended.csv')
patient_df = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Patient_Profile.csv')

#this one has target 
# pp = pd.read_csv('/home/allen/Galva/capstones/capstone2/patient_attendance.csv')

detail = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Health_Camp_Detail.csv')


 

all_camps = pd.concat([event1,event2,event3])
all_camps['Outcome'] = 1


 



train_set = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Train.csv')
test_set = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/test.csv')  







train_IDs = {}
test_IDs = {}
overlap = {}
patient_camp_reg = {}
for i in train_set['Patient_ID'].values:
    if i not in train_IDs:
        train_IDs[i]=1
    else:
        train_IDs[i]+=1
    if i not in overlap:
        overlap[i]=1
    
for i in test_set['Patient_ID'].values:
    if i not in test_IDs:
        test_IDs[i]=1
    else:
        test_IDs[i]+=1
    if i not in overlap:
        overlap[i]=1
    else:
        overlap[i]+=1



repeats = {}
for k in train_set['Registration_Date'].values:
    if k not in repeats:
        repeats[k]=1
    if k == None:
        print('hi')
print(repeats.keys())

one_patient_df =train_set[train_set['Patient_ID'] == 489652 ]

print(one_patient_df)






# train_values= len(train_IDs.keys())
# test_values = len(test_IDs.keys())
# overlap_values = len(overlap.keys())

# print( train_values  , 'is trainset length')
# print(test_values , 'is test length')
# print(overlap_values , 'length of overlap')

# counter = 0
# for key,val in overlap.items():
#     if val > 1:
#         counter +=1
# print(counter) #10,864 patient IDs that have more than 1 count 


# set_test_id = set(test_set['Patient_ID'].values)
# set_train_id = set(train_set['Patient_ID'].values)

# print(len(set_test_id) , len(set_train_id))
# unionz = set_test_id.difference(set_train_id)
# unionz2 = set_train_id.difference(set_test_id) 
# print(len(unionz),len(unionz2))



# set_test_camp = set(test_set['Health_Camp_ID'].values)
# set_train_camp = set(train_set['Health_Camp_ID'].values)

# print(len(set_test_camp) , len(set_train_camp)) 
# unionzz = set_test_camp.union(set_train_camp)
# unionzz2 = set_train_camp.difference(set_test_camp) 
# print(len(unionzz) )
 






















ids = patient_df['Patient_ID'].values
s1 = set(event1['Patient_ID'].values)
s2 = set(event2['Patient_ID'].values)
s3 = set(event3['Patient_ID'].values)
s12 = s1.intersection(s2)
s13 = s1.intersection(s3)
s23 = s2.intersection(s3)
bigu = s1.union(s2)
'''
print(f'length of s1 is {len(s1)}') #length of s1 is 3548
print(f'length of s2 is {len(s2)}') #length of s2 is 6123
print(f'length of s3 is {len(s3)}') #length of s3 is 5340

print(len(s1.union(s2))) #length of union is 8270
print(len(s3.union(s12))) #length of union for all three is 5913
print(len(s1.union(s3))) #length of union for all three is 7727
print(len(s2.union(s3))) #length of union for all three is 9265

print(f'length of BOTH is {len(s12)}') #length of BOTH is 1401 this many did both 
print(f'length of 1_3 is {len(s13)}') #length of 1_3 is 1161
print(f'length of 2_3 is {len(s23)}') #length of 2_3 is 2198
''' 
#looking for filling out stuff overlap : 
# pdf = pd.read_csv('/home/allen/Galva/capstones/capstone2/Patient_and_Target.csv')
# no_age = pdf[pdf['Age'] == "None"] 
#fe0 = pdf[ (pdf['Age'] == "None") and (pdf['Education'] == "None") ]
#print(no_age.describe()) 
#print(pdf['Event1_or_2'].sum())
#combining data frames instead of doing lambdas

# def get_df_counts(columnzs):
#     dict_of_missing = {}
#     for i in columnzs:
#         i_ = pdf[ (pdf['Patient_ID'] != 0 ) & (pdf[i] == "None" ) ]
#         i_2 =set(i_['Patient_ID'].values)
#         dict_of_missing[i]=i_2
#     return dict_of_missing
        
# output = get_df_counts(columnzs= ['Age','Education_Score', 'Income']) 

# age_set = output['Age']
# education_set = output['Education_Score']
# income_set = output['Income']

# all_sets =set()
# all_sets.update(education_set)
# all_sets.update(income_set)
# all_sets.update(age_set)

# def make_sets(x):


# set_outputs = make_sets(x=[age_set,education_set,income_set])

# print(set_outputs)
# def check_sets(x):
#     twos=[]
#     for i in x:
#         for ii in i:
#             if ii not in twos:
#                 twos.append(i)
#     return twos
# two_set = check_sets(x=set_outputs)
# print(len(two_set)) 

# def master_set_of_IDs(): 
#     cols_count= {1:[],2:[],3:all_sets} 
#     for i in 





# heatmap = sns.heatmap(event1.isnull(),yticklabels=False, cbar=False,cmap='viridis')
# print(heatmap)
# 'Online_Follower', 'LinkedIn_Shared', 'Twitter_Shared', 'Facebook_Shared
# 1019, 
# print(patient_df['LinkedIn_Shared'].sum())  848 ,
# print(patient_df['Twitter_Shared'].sum())   813,
# print(patient_df['Facebook_Shared'].sum())    886 
 
#how much overlap in people who shared ? 
#subshare = patient_df[(patient_df['LinkedIn_Shared'] == 1) & (patient_df['Online_Follower'] ==1)]['Patient_ID'].values
#subshare2 = patient_df[(patient_df['Twitter_Shared'] == 1) & (patient_df['Facebook_Shared'] ==1)]['Patient_ID'].values
# print(len(subshare)) #448
# print(len(subshare2)) #442 

#FIRST ATTEMPT WITH LOGISTIC AND GETTING RID OF NANs 
#sus1 = set(subshare).difference(set(subshare2))
#patient_df['City_Type'].fillna(value = 'Z', inplace=True)

# patient_df['Age'].fillna(value = patient_df['Age'].mean(), inplace=True)
#sd={}
#for i in patient_df['Employer_Category'].values:
 #   if i not in sd:
#        sd[i]=1
 #   else:
    #    sd[i]+=1
#

# patient_df1 = patient_df.copy()

# m = {'Software Industry': 1, 'BFSI':2 , 'Education':3 , 'Others':4 , 'Technology': 5, 'Consulting':6 , 'Manufacturing':7, 'Health':8, 'Retail':9,  'Transport': 10 , 'Broadcasting': 11, 'Food':12 , 'Telecom': 13, 'Real Estate':14}
# patient_df1['Job_Type'] = patient_df1['Employer_Category'].map(m) 
# patient_df1['Job_Type'].fillna(value = 9999, inplace=True)

# def impute(x):
#     for _ in x:
#         if _ in [1,2,3,4,5,6,7,8,9]:
#             return 1
#         else:
#             return 0

# patient_df1['Education_Score'].fillna(value=0, inplace=True) 

# patient_df1['Income'].fillna(value=0, inplace=True)


# patient_df1_edu = pd.Series(patient_df1['Education_Score'].values  )
# for i in patient_df1_edu[:10]:
#     print(i)
# patient_df1_edu = patient_df1_edu.transform(lambda x: x ==1 if type(x) is int else 0)
# print(patient_df1_edu[:10])
 
