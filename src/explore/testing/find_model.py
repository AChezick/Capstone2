import pandas as pd 
import numpy as np 
pd.set_option('display.max_columns', None) 
df = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/ready12_24_train.csv')  
from preprocessing import drop_cols , one_hot_encoding , scale
from city_testing import run_test_typeA , run_test_typeC
import itertools as it 
df_no_scale = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/temp_csv/hot_code_NO_scale.csv')
#del df_no_scale['Health_Camp_ID']
del df_no_scale['Category1_x']
del df_no_scale['Category2']
del df_no_scale['Category3']
del df_no_scale['City_Type']
del df_no_scale['Job_Type']
del df_no_scale['online_score']


def run_tests(dataframe):
    '''
    get df, make copy, run tests, store results
    '''
    get =  run_test_typeA(dataframe)


#AttributeError: 'NoneType' object has no attribute 'pop' # ****** Current issue 



    return get

if __name__ =='__main__':
    print(df_no_scale.columns)
    df1 = ['Camp Length', 'Second', 'Third',
    'A', 'C', 'D', 'E', 'F', 'G', '2100', '2.0', '3.0', '4.0', '5.0', '6.0',
    '7.0', '8.0', '9.0', '10.0', '11.0', '12.0', '13.0', '14.0', '9999.0',
    '1', '2', '3', '4', '1036', '1216', '1217', '1352', '1704', '1729',
    '2517', '2662', '23384' ]
    df2= [ '3' ]
    df3=['1036', '1216', '1217', '1352', '1704', '1729',
    '2517', '2662', '23384' ]
    df4=['1']
    df5=[ 'Var3', 'Var4','2.0', '10.0', '11.0','1704', '1729','1','2','3',
       '2517', '2662']
    df6=['4']
    df_list = [df1, df2,df3,df4,df5,df6] # List of dataframe with different columns in each

    resultz_dict = {} 
    cols_dict={}

    for i,ii in enumerate(df_list):
        dataframe = df_no_scale.copy() 
        dataframe.drop(ii, axis=1, inplace=True) # remove columns from list
        get = run_tests(dataframe) 
        i_ = str(i) 
        resultz_dict[i_] = get
        del dataframe 

    #print(resultz_dict)
    
    camp_df = pd.DataFrame.from_dict(resultz_dict, orient='index', columns = ['Health_Camp_ID',
        'False Positive', 'False Negative', 'y_counts_test' , 'y_counts_train' , 'test_size' , 'train_size'])
    #camp_df.to_csv('/home/allen/Galva/capstones/capstone2/data/data_by_feature/results_by_Camp_t2.csv',index=False)

    print(camp_df.head(10))



'''
Results

df1=['2.0', '3.0', '4.0', '5.0', '6.0','7.0', '8.0', '9.0', '10.0', '11.0', '12.0', '13.0', '14.0']
df2=['Var2', 'Var3', 'Var4']
df3=['12.0', '13.0', '14.0', '9999.0','1', '2', '3', '4', '1036', '1216', '1217', '1352', '1704', '1729',
    '2517', '2662', '23384']
df4=['1', '2', '3', '4' ]
df5=['Var2', 'Var3', 'Var4','2.0', '3.0', '4.0', '5.0', '6.0','7.0', '8.0', '9.0', '10.0', '11.0','1704', '1729',
    '2517', '2662']
df6=['5.0'] 

  Health_Camp_ID  False Positive  False Negative  y_counts_test  \
0           6587            2299            2872           4107   
1           6587            2285            2870           4107   
2           6587            2282            2871           4107   
3           6587            2172            2805           4107   
4           6587            2317            2874           4107   
5           6587            2295            2873           4107   

   y_counts_train  test_size  train_size  
0           16427      15056       60222  
1           16427      15056       60222  
2           16427      15056       60222  
3           16427      15056       60222  
4           16427      15056       60222  
5           16427      15056       60222  
'''

'''
df1 = ['C', 'D', 'E', 'F', 'G', '2100', '2.0', '3.0', '4.0', '5.0', '6.0',
    '7.0', '8.0', '9.0', '10.0', '11.0', '12.0']
df2= ['Camp Start Date - Registration Date','1', '2', '3']
df3=['12.0', '13.0', '14.0', '9999.0','1', '2', '3', '4', '1036', '1216', '1217', '1352', '1704', '1729',
    '2517', '2662', '23384']
df4=['Camp Length', 'Second', 'Third',
    'A', 'C', 'D', 'E', 'F', 'G', '2100', '2.0', '3.0', '4.0', '5.0', '6.0',
    '7.0', '8.0', '9.0', '10.0', '11.0', '12.0', '13.0', '14.0', '9999.0',
    '1', '2', '3', '4', '1036', '1216', '1217', '1352', '1704', '1729',
    '2517', '2662', '23384' ]
df5=[ 'Var3', 'Var4','2.0', '10.0', '11.0','1704', '1729','1','2','3',
    '2517', '2662']
df6=['5.0'] 
  Health_Camp_ID  False Positive  False Negative  y_counts_test  \
0           6587            2086            2667           4107   
1           6587            2288            2870           4107   
2           6587            2282            2871           4107   
3           6587             853             575           4107   
4           6587            2288            2869           4107   
5           6587            2295            2873           4107   

   y_counts_train  test_size  train_size  
0           16427      15056       60222  
1           16427      15056       60222  
2           16427      15056       60222  
3           16427      15056       60222  
4           16427      15056       60222  
5           16427      15056       60222 


'''