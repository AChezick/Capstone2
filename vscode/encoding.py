import numpy as np 
import pandas as pd 
pd.set_option('display.max_columns', None) 
df =pd.read_csv('/home/allen/Galva/capstones/capstone2/data/D7.csv')

df['Job_Type'] = df['Job_Type'].apply(lambda x: int(x))

dict_of_cities = {}
for i in df['City_Type'].values:
    if i not in dict_of_cities:
        dict_of_cities[i]=1
    else:
        dict_of_cities[i]+=1
print(dict_of_cities)

df['City_Type'] = df['City_Type'].map(dict_of_cities)
print(df.head())

 
categorical = ['City_Type']
def one_hot_encoding(df, columns = categorical):
    
    for i in columns:
        dummies = pd.get_dummies(df[i], drop_first=True)
        X2 = pd.concat([df, dummies[:]], axis=1)
         
    return X2 
X2 = one_hot_encoding(df, categorical )
print(type(X2) , X2.columns)


