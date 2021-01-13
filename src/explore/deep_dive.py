import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
results_by_city1 = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/results_by_CITY_t1.csv')
results_by_camp1 = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/results_by_Camp_t1.csv')
results_by_city3 = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/results_by_CITY_t3.csv')

pd.set_option('display.max_columns', None) 

def create_cols(dataframe):
    '''
    Create attendance % for Train, Test
    '''

    dataframe['Historic Train'] = dataframe['train_size'].apply(lambda x: x *.2)  
    dataframe['Historic Test'] = dataframe['test_size'].apply(lambda x: x *.2) 

    dataframe['Ratio'] = dataframe['False Negative']/dataframe['False Positive']
    dataframe['Train Percent Atttends'] = dataframe['y_counts_train']/dataframe['train_size']
    dataframe['Test Percent Atttends'] = dataframe['y_counts_test']/dataframe['test_size']

    dataframe['Model - Historic Train'] = dataframe['Historic Train'] - dataframe['y_counts_train']
    dataframe['Model - Historic Test'] = dataframe['Historic Test'] - dataframe['y_counts_test']
                                                    # 20   - 40 = -20 #negative means they UNDER predicted meaning not enough supplies 
                                                    #20    - 15 = 5  # positive means they OVER predicted meansing too much suppplies 
    return dataframe
    '''
    A. multiply [Historic Attends] by actual [train_size] , [test_size] = number of Historic misses # The easy cheap solution
    B. multiply [Percent Attends] by actual [train_size] , [test_size] = number of predicted misses # The models solution 
    C. Obtian difference  = A - B 
    '''



if __name__=='__main__':
    df_city1 = create_cols(results_by_city1)
    df_city3 = create_cols(results_by_city3)
    df_camp1 = create_cols(results_by_camp1)
    print(df_camp1)