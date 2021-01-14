import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
results_by_city1 = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/results_by_CITY_t1.csv')
results_by_camp1 = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/results_by_Camp_t1.csv')
results_by_city3 = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/results_by_CITY_t3.csv')
results_by_camp3 = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/results_by_Camp_t3.csv')
results_XG1 = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/data_by_feature/XG_CITY_t1.csv')
results_XG2 = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/data_by_feature/XG_CITY_t2.csv')
pd.set_option('display.max_columns', None) 

def create_cols(dataframe):
    '''
    Create features to compare model results among specific camps / city locations
    '''

    dataframe['Historic Train'] = dataframe['train_size'].apply(lambda x: x *.2)  #20% historic attendance rate
    dataframe['Historic Test'] = dataframe['test_size'].apply(lambda x: x *.2) 

    dataframe['Ratio'] = dataframe['False Negative']/dataframe['False Positive']
    dataframe['Train Percent Atttends'] = dataframe['y_counts_train']/dataframe['train_size']
    dataframe['Test Percent Atttends'] = dataframe['y_counts_test']/dataframe['test_size']

    dataframe['True Y- Historic Train'] =  dataframe['y_counts_train'] - dataframe['Historic Train'] 
    dataframe['Their Error'] =  dataframe['y_counts_test'] - dataframe['Historic Test']  
                                       
    dataframe['Total Model Error'] = dataframe['False Positive'] + dataframe['False Negative']
    dataframe['Total Y'] = dataframe['y_counts_train'] + dataframe['y_counts_test']
    tme = dataframe['Total Model Error'].values 
    ht = dataframe['Their Error'].values

    combo = list(zip(tme,ht))
    model_minus_historic = []
    for i in combo:
        if i[1] > 0 and i[0] < 0:
            get = i[0] - i[1] + .069 
            model_minus_historic.append(get)
        else:
            get = i[0] - i[1]
            model_minus_historic.append(get)
    

    dataframe['Model - Historic Test'] = model_minus_historic 
                                                # Myerror 20 - their error 40  = -20 people difference = negative value means I did better like Golf
                                                # My error 15  - their error - 15 =  0 difference = same
                                                # Myerror 40 - their error 20  = 20 people difference = positive value means I did worse like Golf
                                                # Myerror 40 - their error -20  = 60 people difference = positive value means I did worse like Golf
                                                # Myerror -40 - their error -20  = -20 people difference = negative value and I did worse  

    return dataframe

if __name__=='__main__':
    df_xg_city1 = create_cols(results_XG1)
    df_xg_city2 = create_cols(results_XG2)
    df_city1 = create_cols(results_by_city1)
    df_city3 = create_cols(results_by_city3)
    df_camp1 = create_cols(results_by_camp1)
    df_camp3 = create_cols(results_by_camp3)
    print(df_xg_city2) 

    print( df_xg_city2['Total Model Error'].sum() , df_xg_city2['Their Error'].sum()) #
    

    # print( df_city3['Total Model Error'].sum() , df_city3['Their Error'].sum()) #
    # print( df_city1['Total Model Error'].sum() , df_city1['Their Error'].sum()) 