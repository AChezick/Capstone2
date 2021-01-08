
import numpy as np 
import pandas as pd 
pd.set_option('display.max_columns', None) 
df =pd.read_csv('/home/allen/Galva/capstones/capstone2/data/D7.csv')
df2 = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/ready12_24_train.csv') 

drop_thez=[ 'Patient_ID_x', 'Health_Camp_ID', 'Registration_Date', 'Category1_y','Camp_Start_Date2', 
    'Camp_End_Date2', 'patient_event', 'Unnamed: 0_x','Unnamed: 0.1_x', 'Online_Follower_x', 'First_Interaction'
    ,'Employer_Category' , 'Event1_or_2_x' ,'Category1_y', 'Unnamed: 0_y', 'Unnamed: 0.1_y', 'Patient_ID_y',
    'Online_Follower_y', 'Event1_or_2_y','Health Score', 'Camp_length',
    'Number_of_stall_visited', 'Last_Stall_Visited_Number']

df3 = df2.drop(drop_thez, axis =1)


camp_info = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Health_Camp_Detail.csv')
fuck = {}
for i in camp_info['Category3'].values:
    if i not in fuck:
        fuck[i]=1
    else:
        fuck[i] +=1
print(fuck)


def one_hot_encoding(dataframe, columns):
    '''
    Hot encoding of categorical columns 
    '''
    for i in columns:
        dummies = pd.get_dummies(dataframe[i], drop_first=True)
        hot_df = pd.concat([dataframe, dummies[:]], axis=1)

    #hot_df = hot_df.drop(columns,axis=1)
    return hot_df 


df_encode = one_hot_encoding(df3, columns = ['Category1_x'])

df_encode2 = one_hot_encoding(df_encode, columns = ['City_Type'])
print(df_encode2.columns)
df_encode3 = one_hot_encoding(df_encode2, columns = ['Category2']) 
print(df_encode3.columns)
df_encode4 = one_hot_encoding(df_encode3, columns = ['Category3']) 
print(df_encode4.columns)
df_encode5 = one_hot_encoding(df_encode4, columns = ['Job_Type']) 

df_encode6 = one_hot_encoding(df_encode5, columns = ['online_score']) 

print(df_encode6.columns)






















''' 
df2['Job_Type'] = df2['Job_Type'].apply(lambda x: int(x))

dict_of_cities = {}
for i in df['City_Type'].values:
    if i not in dict_of_cities:
        dict_of_cities[i]=1
    else:
        dict_of_cities[i]+=1

df2['City_Type'] = df2['City_Type'].map(dict_of_cities)


drop_thez=[ 'Patient_ID_x', 'Health_Camp_ID', 'Registration_Date', 'Category1_y','Camp_Start_Date2', 
    'Camp_End_Date2', 'patient_event', 'Unnamed: 0_x','Unnamed: 0.1_x', 'Online_Follower_x', 'First_Interaction'
    ,'Employer_Category' , 'Event1_or_2_x' ,'Category1_y', 'Unnamed: 0_y', 'Unnamed: 0.1_y', 'Patient_ID_y',
    'Online_Follower_y', 'Event1_or_2_y','Health Score', 'Camp_length',
    'Number_of_stall_visited', 'Last_Stall_Visited_Number']
df2 = df2.drop(drop_thez, axis =1)
 
categorical = ['City_Type']
def one_hot_encoding(df2, columns = categorical):
    
    for i in columns:
        dummies = pd.get_dummies(df2[i], drop_first=True)
        X2 = pd.concat([df2, dummies[:]], axis=1)
         
    return X2 
X2 = one_hot_encoding(df2, categorical )
print(type(X2) , X2.columns)

# 'City_Type','Category1_x','Category2','Category3', 'Job_Type', 
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder

def one_hot2(x,colz):

    cities = df2['City_Type'].unique
    print(cities)
    # encoder = LabelEncoder()
    # citi_labels =  encoder.fit(cities)
    # print(citi_labels)
    # l = len(citi_labels)
    # print(l)
    # encoder2 = OneHotEncoder(sparse=False)
 
    # citi_labels = citi_labels.reshape( (l,1) )  # might just hard code this? 
    # encoder2.fit_transform(citi_labels)
    # print(encoder2)
    return x

outputz = one_hot2(x=df2, colz = ['City_Type'] )
print(outputz) 

''' 