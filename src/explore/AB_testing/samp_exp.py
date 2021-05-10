import numpy as np
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
    accepts an payoff % as input for a bandit 
    creates array of zeros and ones to be chosen 
    '''
    onez = np.ones( int(x * 100))
    zeroz= np.zeros(100-int(x * 100))
    bandit = np.hstack((onez,zeroz))
    return bandit

def get_bandits(xx ,nn,ratez):
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
        
    check_ = x.get(check_key)             
 
    if check_key == 'a': #check which bandit is biggest 
        get = random.choice(get_a) #pull bandit, store output in get 
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

def experiment_numerical(params):
    exp_resultz = {}
    nn=1
    ratez = params[1]
    #xx = {'a':[1.0, .5, 2 ] ,'b':[1.0, .5, 2 ] , 'c':[1.0,.5, 2]  }   
    x = params[0].copy() 

    for i in range(1,500): 
        i_ =i -1

        if i>2:
            exp_resultz.pop(i_ ) #remove last bandit , helps with view of current printout 

        output = get_bandits( x ,nn, ratez) #sending dict to bandit funt
        x = output #replace x with updates from get bandit 
        nn+=1  #update expermential count 
        exp_resultz[i]=[x] # add new pay_out version to dict

        if nn == 500:
            ap,bp,cp = round((x['a'][0]/(x['a'][2])),2), round((x['b'][0]/(x['b'][2])),3) ,round((x['c'][0]/(x['c'][2])),3)
            indv_wins =  x['a'][0] ,x['b'][0] , x['c'][0]
            total_wins = x['a'][0] + x['b'][0] + x['c'][0]
            return exp_resultz #indv_wins #['actual rates',ap,bp,cp ]#, 'total_wins',total_wins   ] # 

if __name__ == '__main__': 
    # print(funct_class(x=[n_bandits,win_rates,starting_win_rates]))
    print( experiment_numerical(params = [ {'a':[1.0, .5, 2 ], 'b':[1.0, .5, 2 ], 'c': [1.0, .5, 2 ]} , [ 0.03, 0.05, 0.08] ] ))
  
''' 
Dep_Vars = win_rate & number of wins
 
Quetions:
-How does changing the number of bandits impact winrate / number of wins?
-How does changing the pay_out_% impact winrate / number of wins?
-How does changing the starting distribution impact winrate / number of wins?

Controls:
n_trials, win_rates, 
'''

# Create a test file with a function for each above
# Graphing outputs
# 
#4/24 trying to pass in params for doing experiments that cover differ inputs
#### Not allowing me to pass in a dict of values of initial inputs ? 