
# 5/10/21
'''
AB_testing tab is currently under construction.

by_date_camps.py is the main testing file
postHOCAB.py is the modeling file 
cap_ab_tesing.py is the thompson samping implementation 


This script :
1. Takes in test DF
2. Each model's probabilities for patient attendance will be added as columns to the test_DF
3. Test DF will be sent back for parsing in AB testing. 
'''



import numpy as np
from numpy.lib.type_check import _nan_to_num_dispatcher
import pandas as pd 
import scipy as scipy 
import scipy.stats as stats 
import time as time 
import timeit 
import random 
from scipy.stats import beta

from random import choice   
 

def get_bandits(x ,nn,ratez):
    '''
    5/24 Updates
                y_target  Patient_ID  SVC  svc_preds
      row_var = [0.0        489652     0.0   0.190603] row_var == ratez 

    - I think instead create_bandit , get_abc would point to that model in the row_var
    - Here is where I can create 'consenious bandit' = majority vote 
                    <need 3 or 5 bandits or random.chocie for tie breaker>

    x_former_DF_row_var = {'svc': [1, 3, 2], 'log': [4, 3, 2], 'y_target': [0, 0, 1]}
    '''

    xx= ratez.copy()  
     
    get_a = x['a'][nn] # using nn_trial as index
    get_b = x['b'][nn]
    get_c = x['c'][nn] #+ ratez['svc'] # if sum >= majority, 1 else 0 
    # get_d = x['d'][nn]
    # get_e = x['e'][nn]
    best_rate = 0 
    check_key = 'z' 
    trial = nn   

    for key,value in ratez.items() : # for each bandit calc new bandit win percentage 
 
        if trial%2==0:
            wins,rate,plays = value[0],value[1],value[2] 
            winz_rate = np.random.beta(wins,plays)
            new_rate = round(winz_rate,2)

            if new_rate < rate:  
                new_rate = rate - .0005 #impose small pentality 
                #print(new_rate, key, 'for sanity this is new_rate & key in x.items()') 

        if trial%2 !=0:
            wins,rate,plays = value[0],value[1],value[2] 
            winz_rate = np.random.beta(wins,plays)
            new_rate = round(winz_rate,2)  
             
        if new_rate > best_rate: #get key for biggest win_rate 
            best_rate = new_rate
            check_key = key 
           
        xx[key]=[wins,new_rate , plays ] #building new list to iterate     
        
    check_ = xx.get(check_key)     #get key to access values of best win rate for pull         
 
    if check_key == 'a': # if model 'a' is the best so far check it's prediction 
        get = get_a # 'pull' bandit will be get model's prediction 
        check_[0] += get # updating 2nd value for bandit A  
        check_[2] +=1 
        xx[check_key] = check_ 
        trial+=1
        return xx 
  
    if check_key=='b':
        get = get_b
        check_[0] += get # updating 2nd value for bandit B
        check_[2] +=1 
        xx[check_key] = check_ 
        trial += 1 
        return xx 

    # if check_key=='d':
    #     get = get_d
    #     check_[0] += get # updating 2nd value for bandit B
    #     check_[2] +=1 
    #     xx[check_key] = check_ 
    #     trial += 1 
    #     return xx 
    # if check_key=='e':
    #     get = get_e
    #     check_[0] += get # updating 2nd value for bandit B
    #     check_[2] +=1 
    #     xx[check_key] = check_ 
    #     trial += 1 
    #     return xx 
    else:
        get =  get_c 
        check_[0] += get 
        check_[2] +=1 
        xx[check_key] = check_ 
        trial+=1 
        return xx

    return xx    

def parse(df):
    '''
    input: data Frame
    actions: del un needed columns 
    returns: df
    '''
    to_del =  ['Unnamed: 0', 'Health_Camp_ID', 'Var1', 'Var2', 'Var3',
       'Var4', 'Var5',  'Camp Length', 'BFSI', 'Broadcasting',
       'Consulting', 'Education', 'Food', 'Health', 'Manufacturing', 'Others',
       'Real Estate', 'Retail', 'Software Industry', 'Technology', 'Telecom',
       'Transport', 'A', 'C', 'D', 'E', 'F', 'G', 'Second', 'Third',
       '1', '2', '3', '4']
    df_ = df.drop( to_del , axis=1)

    df_['a'] = df['SVC']
    df_['b'] = df['XG']
    df_['c'] = df['log_preds'] 
    #df_['d'] = df['SVC2'] 
    # print(df_.head(2))
    return df_ 

def experiment_numerical(dataframe,params={'a':[1.0, .5, 2 ] , 'b':[1.0, .5, 2 ] , 'c': [1.0, .5, 2 ]}): # , 'd': [1.0, .5, 2 ], 'e': [1.0, .5, 2 ]
    '''
    -recieves a dict & dict_of_results 
    -iterates through DF and updates dict based on each trial 
    - win rates will need to be separated 
    '''
    exp_resultz = {}
    nn=1
    
    #ratez = [v[1] for k,v in params.items()] #previous win rates * might need to pass all of params
    ratez2 = params 
    #print(dataframe)
    for index in range(len(dataframe.keys())): # iterate through patients/trials 

        i_ = index-1
        if i_>2:
            exp_resultz.pop(i_)

        
        x = dataframe.to_dict() # 5/25 maybe send entire dict and use trial number to point to index for patient_ID
        output = get_bandits( x ,nn, ratez2) #sending dict, trial_num, pay_outs --> to bandit funt
        '''
        Question: How does bandit play? 
        Ans:
        For each patient/trial the patient row is assigned to row_var 
                    y_target  Patient_ID  SVC  svc_preds
        row_var = [0.0        489652     0.0   0.190603] row_var == ratez 
        Send row_var along with the dictionary (*below) to get_bandits 
        *dict ={'xg':[1.0, .5, 2 ], 'svc':[1.0, .5, 2 ] , 'log': [1.0, .5, 2 ], 'avg':[1.0, .5, 2 ]} 
        '''

        ratez2 = output #replace x with updates from get bandit 
        nn+=1  #update expermential count 
        exp_resultz[index]=output # add new pay_out version to dict
        
        if nn == len(dataframe):
            ap,bp,cp = round((ratez2['a'][0]/(ratez2['a'][2])),2), round((ratez2['b'][0]/(ratez2['b'][2])),3) ,round((ratez2['c'][0]/(ratez2['c'][2])),3)
            #dp , ep = round((ratez2['d'][0]/(ratez2['d'][2])),2) , round((ratez2['e'][0]/(ratez2['e'][2])),2)
            indv_wins =  ratez2['a'][0] ,ratez2['b'][0] , ratez2['c'][0]
            total_wins = ratez2['a'][0] + ratez2['b'][0] + ratez2['c'][0]
            return exp_resultz  

if __name__ == '__main__': 
    df = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/temp_csv/thomps2.csv') 
    print(df)
    parsed = parse(df) 
    print(experiment_numerical(parsed , params={'a':[1.0, .5, 2 ] , 'b':[1.0, .5, 2 ] , 'c': [1.0, .5, 2 ]})) # , 'd': [1.0, .5, 2 ], 'e': [1.0, .5, 2 ]
    #print( experiment_numerical(params = [ {'a':[1.0, .5, 2 ], 'b':[1.0, .5, 2 ], 'c': [1.0, .5, 2 ]} , [ 0.03, 0.05, 0.08] ] ))
  
''' 
5/6/21

Editing this file to get input from modelings 

input = [dict , dataframe] 
-dict ={'xg':[1.0, .5, 2 ], 'svc':[1.0, .5, 2 ] , 'log': [1.0, .5, 2 ], 'avg':[1.0, .5, 2 ]} 
-df = [patientID , campID, xgP,xg_p , svcP,svc_p, knnP,knn_p , avgP,avg_p ]
patient_1
patient_2
.
.
.
len(df) = num trials


5/24
- Create a parsing function to clean DF and then send through testing pipline 

'''

 