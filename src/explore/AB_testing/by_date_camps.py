
# 5/10/21
'''
AB_testing tab is currently under construction.

by_date_camps.py is the main testing file
postHOCAB.py is the modeling file 

 

This script :
1. Creates a list of Health Camps by Date, determines which health camp patients
   can be used to train prediction models
2. Two data frames are made by subsetting the main testing file 
3. The Data frames are sent out for Thompson Samping 
4. Bandit (the models) probabilities are updated as they are played 
'''
 

from cap_ab_testing import experiment_numerical
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
from preprocessing_formatting import drop_cols , one_hot_encoding , scale
from postHOCAB import run_tests 


pd.set_option('display.max_columns', None) 
dataframe = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/temp_csv/ab_df.csv') 
ids = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/testing/x1.csv')
df_withdates = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/placeholder/df_withdates.csv') 
df_blank = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/ready12_24_train.csv') 

def edit_df():
    '''
    5/20 
    This is being edited since new csv was made. 
    '''

    df = dataframe.drop(['City_Type2_x','Job Type_x','Category 2','Category 3','Category 1', 'online_score'],axis=1) 
    return df

def sep_by_date(df_encode):
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
            if start > end2: # check if start date camp_1 is after end date camp_2
                d_time[k].append(kk)

    ans = [(k,v) for k,v in d_time.items() ] 
    return ans


def parse_results():
    '''
    reformat modeling results for easier parsing in AB testing
    '''
    #What to send to do_ab testing? 

    return None

def create_df(df, keys): #5/7 Main 'Function for AB pipeline' 
    '''
    Create train,test DFs from sorted camps

    Each Test Round:
    Send test,train DF to models for training/results
    Send results for AB testing
    Keep track of results & Update models 
    '''
    del df['Start']
    del df['End']
    

    df1 = df.copy()
    df2 = df.copy() 
     
    model_bandits ={'xg':[1.0, .5, 2 ], 'svc':[1.0, .5, 2 ] , 'log': [1.0, .5, 2 ], 'avg':[1.0, .5, 2 ]}
    model_check = [] #Del this line eventually / EXCHANGE FOR making pickle files 
    win_rates = [v[2] for k,v in model_bandits.items()]
    for item in keys[3:5]: # [(6530, ['NA']), (6560, ['NA']), (6544, ['NA', 6530, 6560]), (6561, ['NA', 6530, 6560, 6544]), (6585, ['NA', 6530, 6560, 6544])
        if len(item[1]) <=1: #
            break
        else:
            iD = item[0] # Camp_ID
            camps = item[1][1:] #list of camps for training 

            test_df = df1[df1['Health_Camp_ID'] == iD ]  
            train_df = df2.loc[ df2['Health_Camp_ID'].isin(camps)  ]  

            
            do_modeling = run_tests(test_df,train_df) # This should be parsed before sending to do_testing
            model_check.append(do_modeling)
            # do_modeling.to_csv('/home/allen/Galva/capstones/capstone2/src/explore/temp_csv/thomps.csv') # Help with next phase 
            # parser = parse_results(do_modeling)
            
            #do_testing = experiment_numerical(model_bandits , parser )

            # Will then need to update model_bandits 

        return model_check , 'SUCCESSSSSS'




if __name__ == '__main__':
    step1 = edit_df()
    # step1.to_csv('/home/allen/Galva/capstone/capstone2/src/explore/AB_testing/for_ab_modeling.csv')
    step2 = sep_by_date(step1)
   # step3 = step2.sort(key = lambda x : x[1]) # wont need this for final modeling 
    step3 = create_df(step1 , step2)
    print(step3)
    #for item in step3: # For each camp to predict and the camps to train for that prediction 
        # send item through create_df 
       # print(item)


'''
5/24
- Parser function will need to be completed 
---currently the beta is being done in cap_ab_testing 
- Same csv made to expeidate the next steps 

'''
   
