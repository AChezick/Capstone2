import pandas as pd 
import numpy as np 
pd.set_option('display.max_columns', None) 

df = pd.read_csv('main_df.csv')
df_city = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/D7.csv')


def impute_city(x):
    dict_of_cities = {}
    for i in df_city['City_Type'].values:
        if i not in dict_of_cities:
            dict_of_cities[i]=1
        else:
            dict_of_cities[i]+=1
    print(dict_of_cities.items())

    df['City_Type'] = df_city['City_Type'].map(dict_of_cities)
    return df 

 


if __name__ == '__main__':
    print(df.head() )
    df_and_city = impute_city(x=df)
    print(df_and_city['City_Type'])

