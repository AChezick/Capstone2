import pandas as pd 
import numpy as np 
pd.set_option('display.max_columns', None) 

#df = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/ready12_24_train.csv') 

def create_new_cols(x):
    '''
    create new columns
    '''
    x.delta_first_reg = x.delta_first_reg.astype(int)
    x.interaction_regreister_delta = x.interaction_regreister_delta.astype(int)
    x.delta_first_start = x.delta_first_start.astype(int)
    x.Camp_Length = x.Camp_Length.astype(int)
    return x 

 

if __name__ == '__main__':
    #cols = create_new_cols(df) 

    print(create_cols())
 