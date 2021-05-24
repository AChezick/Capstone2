
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

#r1,r2,r3 = .13 , .02 , .08  

def create_bandit(x):
    '''
    accepts a payoff % as input for a bandit 
    creates array of zeros and ones to be chosen 
    '''
    onez = np.ones( int(x * 100))
    zeroz= np.zeros(100-int(x * 100))
    bandit = np.hstack((onez,zeroz))
    return bandit

def get_bandits(xx ,nn,ratez):
    '''
    5/24 Updates
                y_target  Patient_ID  SVC  svc_preds
      row_var = [0.0        489652     0.0   0.190603] row_var == ratez 
    '''

    x=xx.copy() # needs to be automated 
    r1,r2,r3 = ratez[0],ratez[1],ratez[2]
    get_a = create_bandit(r1)
    get_b = create_bandit(r2)
    get_c = create_bandit(r3)
    best_rate = 0 
    check_key = 'z' 
    trial = nn   

    for key,value in x.items() : # for each bandit calc new bandit win percentage 
 
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
           
        x[key]=[wins,new_rate , plays ] #building new list to iterate     
        
    check_ = x.get(check_key)     #get key to access values of best win rate for pull         
 
    if check_key == 'a': 
        get = random.choice(get_a) #pull bandit will be get output from dataframe
        # get = find patient ID in DF and point to model's prediction 0/1 

        check_[0] += get # updating 2nd value for bandit A  
        check_[2] +=1 
        x[check_key] = check_ 
        trial+=1
        return x 
  
    if check_key=='b':
        get = random.choice(get_b)
        check_[0] += get # updating 2nd value for bandit B
        check_[2] +=1 
        x[check_key] = check_ 
        trial += 1 
        return x 

    else:
        get = random.choice(get_c)
        check_[0] += get 
        check_[2] +=1 
        x[check_key] = check_ 
        trial+=1 
        return x

    return x    

def parse(df):
    '''
    input: data Frame
    actions: del un needed columns 
    returns: df
    '''
    to_del =  ['Unnamed: 0', 'Unnamed: 0.1', 'Health_Camp_ID', 'Var1', 'Var2', 'Var3',
       'Var4', 'Var5',  'Camp Length', '1036', '1216', '1217',
       '1352', '1704', '1729', '2517', '2662', '23384', 'BFSI', 'Broadcasting',
       'Consulting', 'Education', 'Food', 'Health', 'Manufacturing', 'Others',
       'Real Estate', 'Retail', 'Software Industry', 'Technology', 'Telecom',
       'Transport', 'A', 'C', 'D', 'E', 'F', 'G', '2100', 'Second', 'Third',
       '1', '2', '3', '4']
    df_ = df.drop( to_del , axis=1)
    return df_ 

def experiment_numerical(dataframe,params):
    '''
    -recieves a dict & dict_of_results 
    -iterates through DF and updates dict based on each trial 
    - win rates will need to be separated 
    '''
    exp_resultz = {}
    nn=1
    
    ratez = [v[2] for k,v in params.items()] #previous win rates 
    x = dataframe.copy() 

    for i in dataframe: # iterate through patients/trials 
                        # send entire row? or pointers to bandit function 
        i_ =i -1

        if i>2:
            exp_resultz.pop(i_ ) #remove last bandit , helps with view of current printout 

        output = get_bandits( x ,nn, ratez) #sending dict, trial_num, pay_outs --> to bandit funt
        '''
        Question: How does bandit play? 
        Ans:
        For each patient/trial the patient row is assigned to row_var 
                    y_target  Patient_ID  SVC  svc_preds
        row_var = [0.0        489652     0.0   0.190603] row_var == ratez 
        Send row_var along with the dictionary (*below) to get_bandits 
        *dict ={'xg':[1.0, .5, 2 ], 'svc':[1.0, .5, 2 ] , 'log': [1.0, .5, 2 ], 'avg':[1.0, .5, 2 ]} 
        '''

        x = output #replace x with updates from get bandit 
        nn+=1  #update expermential count 
        exp_resultz[i]=[x] # add new pay_out version to dict

        if nn == len(dataframe):
            ap,bp,cp = round((x['a'][0]/(x['a'][2])),2), round((x['b'][0]/(x['b'][2])),3) ,round((x['c'][0]/(x['c'][2])),3)
            indv_wins =  x['a'][0] ,x['b'][0] , x['c'][0]
            total_wins = x['a'][0] + x['b'][0] + x['c'][0]
            return exp_resultz  

if __name__ == '__main__': 
    df = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/temp_csv/thomps.csv') 
    print(df)
    print(parse(df))
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

 