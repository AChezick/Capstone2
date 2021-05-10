import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from preprocessing import drop_cols , one_hot_encoding , scale
pd.set_option('display.max_columns', None) 
df_withdates = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/placeholder/df_withdates.csv') 
df_blank = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/ready12_24_train.csv') 
# Dataframe = # Find a csv with scaled/ data ready 

start = df_withdates['Camp_Start_Date2'].values
end = df_withdates['Camp_End_Date2'].values

def edit_df():
    '''
    create new cols 
    '''
    data = drop_cols(df_blank) # drop cols
    df_encode = one_hot_encoding(data, columns = ['City_Type2_x','Job Type_x','Category 2','Category 3','Category 1', 'online_score'])
    df_encode['Start']=start
    df_encode['End'] = end
    return df_encode 

def sep1(df_encode):
    '''
    input = df
    actions = Create dict of camp_IDs, compare each start date to end_date
    create tupple of camp_ID and other Campsz that end before camp_ID starts (IDs that can be used for model training)
    '''
    start = df_withdates['Camp_Start_Date2'].values
    end = df_withdates['Camp_End_Date2'].values
    camp_id = df_encode['Health_Camp_ID'].values 
    unique={}
    d_time = {}
    for item in list(zip(start,end, camp_id)):
        if item not in unique:
            unique[item[2]] =item

    for item in camp_id:
        if item not in d_time:
            d_time[item] = ['NA']

    for k,v in unique.items():
        
        start = v[0]
        end = v[1]
        for kk,vv in unique.items():
            start2 = vv[0]
            end2 = vv[1]
            if start > end2: # check if start date if after end date
                d_time[k].append(kk)

    ans = [(k,v) for k,v in d_time.items() ] 
    return ans


def parse_results():
    '''
    reformat modeling results for easier parsing in AB testing
    '''
    #What to send to do_ab testing? 

    return None

def create_df(keys):
    '''
    Create train,test DFs from sorted camps
    Send to testing out for AB testing
    Keep track of results
    '''
    model_bandits ={'xg':[1.0, .5, 2 ], 'svc':[1.0, .5, 2 ] , 'log': [1.0, .5, 2 ], 'avg':[1.0, .5, 2 ]}
    for item in keys:

        item[0] # Camp_ID
        item[1] #list of camps for training 
        test_df = dataframe[dataframe['Camp_ID'] == ]  
        train_df = dataframe[dataframe['Camp_ID']== []]

        do_modeling = pointer_to_models(test_df,train_df) # This should be parsed before sending to do_testing

        #parser = pointer to parser(do_modeling)
        do_testing = pointer_to_AB(model_bandits , )






    return None




if __name__ == '__main__':

    step1 = edit_df()
    step2 = sep1(step1)
    sort_step2 = step2.sort(key = lambda x : x[1])
    for item in sort_step2:
        print(item[0], len(item[1]) -1  )


'''
sort_step2 



'''
   
