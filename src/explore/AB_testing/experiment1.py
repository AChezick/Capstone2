# import numpy as np
# import pandas as pd 
# import scipy as scipy 
# import scipy.stats as stats 
# import time as time 
# import timeit 
# import random 
# from scipy.stats import beta
# from random import choice   
from samp_exp import experiment_numerical    
#from bayesian_bandit import experiment  , output  

#numeric1,numeric2,numeric3,numeric4,numeric5,numeric6 = experiment_numerical(),experiment_numerical(),experiment_numerical(),experiment_numerical(),experiment_numerical(),experiment_numerical()
#reg1,reg2,reg3,reg4,reg5,reg6 = experiment() ,experiment() , experiment() , experiment() , experiment() ,experiment()


#params = ({1: [{'a': [1.0, 0.05, 2], 'b': [1.0, 0.81, 3], 'c': [1.0, 0.42, 2]}], 499: [{'a': [3.0, 0.1, 21], 'b': [10.0, 0.24, 38], 'c': [226.0, 0.35, 446]}]}, ['actual rates', 0.14, 0.263, 0.507, 'total_wins', 239.0])
#params = [ {'a':[1.0, .5, 2 ], 'b':[1.0, .5, 2 ], 'c': [1.0, .5, 2 ]} , [ 0.03, 0.05, 0.08] ]

##
# Going into samp_exp ---> experiment_numerical(params = [ {'a':[1.0, .5, 2 ], 'b':[1.0, .5, 2 ], 'c': [1.0, .5, 2 ]} , [ 0.03, 0.05, 0.08] ]
##

def make_df(x):
    '''
    take dict resuts and make DF 
    Needs to account for multiple bandits 
    '''
    print(x ,'x is being PRINTED', type(x))
    for item in x:
        print(type(item), 'type item here')
    # will need to count keys to make sets of arrays for DF
    # column names will be letter_of_bandit [a,b,c..] + 'str' [wins, rate,plays]  
    # rows will be values from df   
    '''
    item = {'a': [13.0, 0.12, 89], 'b': [13.0, 0.09, 95], 'c': [68.0, 0.2, 321]}
    get_item.keys() into list of keys = (a,b,c ...)
    building len(get_item.keys()) sets of 3 arrays -> v[0],v[1],v[2] 
    --for the Column Values 
    values
    '''
    return 'hi'

def parse(params):
    resultz={}
    for i in range(2):
        get_numeric = experiment_numerical(params) 
        #get_thomps = output(params)
        make_print  = make_df(get_numeric.get(499)) # this will

        resultz[i]= get_numeric.get(499)  #,get_thomps]) 

    print(resultz.keys(), 'printLINE --- Experiment being done **'  )
    return resultz 
 
if __name__ == '__main__': 
    params = [ {'a':[1.0, .5, 2 ], 'b':[1.0, .5, 2 ], 'c': [1.0, .5, 2 ]} , [ 0.03, 0.06, 0.12] ]
    x = .02
    # resultzz = {'a0':[] , 'a1':[], 'a2':[] ,'b0':[] , 'b1':[] , 'b2':[],'c0':[] , 'c1':[] , 'c2':[] } #would fail once we edit the number of bandits
    for i in range(1):
        x += .05
        params_= [ params[0] , [ params[1][0] +x , params[1][1] +x  , params[1][2] +x ] ] 
        get_exp = parse(params_ ) 
        resultzz[str(x) + 'Winrate'] = get_exp  # 5/4 this might be changed based on what we are doing currently 

        #print(get_exp.values() , len(get_exp.values())) 
        for item in get_exp.values():
            print(item , 'item-----------------------------')
            make_df here ######consult capstone2 

            #for thing in item:
                #print(thing.values(), type(thing), 'thingggggggggggggggggggggggggggggg')





                 # running totals /// Make Data Frame from each trial 
                # 'a0'[0].append(i.a[0]),
                # 'a1'[1],'a2'[2]  
                # 'b0'[0],'b1'[1],'b2'[2] = i.b[0]  
                # 'c0'[0],'c1'[1],'c2'[2] = i.c[0]  

      #print(parse(params = [ {'a':[1.0, .5, 2 ], 'b':[1.0, .5, 2 ], 'c': [1.0, .5, 2 ]} , [ 0.03, 0.06, 0.12] ]))

# 5/4/21      
## Trying to build data frame from experimental results  
'''
Dep_Vars = win_rate & number of wins
 
Quetions:
-How does changing the number of bandits impact winrate / number of wins?
-How does changing the pay_out_% impact winrate / number of wins?
-How does changing the starting distribution impact winrate / number of wins?

Controls:
n_trials
'''
# How many trials for each 'run' ? 
##try 10, average results, complete T-test 

# Initial Test
'''
dict where 
keys are strings of rate being applied
values is a dict
the dict has the results of 0/10 trials

{'0.07Winrate':

{0: [{'a': [5.0, 0.1, 56], 'b': [14.0, 0.09, 125], 'c': [67.0, 0.16, 324]}], # A list with a dict with keys for each bandit
1: [{'a': [2.0, 0.06, 38], 'b': [6.0, 0.07, 64], 'c': [80.0, 0.16, 403]}], 
2: [{'a': [5.0, 0.09, 49], 'b': [5.0, 0.11, 50], 'c': [83.0, 0.18, 406]}], 
3: [{'a': [7.0, 0.06, 72], 'b': [7.0, 0.14, 64], 'c': [71.0, 0.16, 369]}],
9: [{}] 

'nextWinrate': repeat,

'nextWinrate: repeat 

} 

-suppose could write a detangle alog 

5/4 thoughts on detangle



'''

