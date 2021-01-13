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


# def create_cols():
#     '''
#     function to create columns / data frames 
#     '''
#     camp_id= [6578, 6532, 6543, 6580, 6570, 6542, 6571, 6527, 6526, 6539, 6528,
#      6555, 6541,6523, 6538, 6549, 6586, 6554, 6529, 6540, 6534, 6535, 6561, 6585, 
#      6536, 6562, 6537, 6581, 6524, 6587, 6557, 6546, 6569, 6564, 6575, 6552, 6558, 
#      6530, 6560, 6531, 6544, 6565, 6553, 6563]



#     get = list(zip(camp_id,camp_list))
#     return get 

#     for item in camp_id:
#         get = df[df['COI'] == i]
#         scale_get
#         test_get 
#         print(results)

if __name__ == '__main__':
    #cols = create_new_cols(df) 

    print(create_cols())

# will need to go back and complete for the test.csv file  <which will NOT be used in initital modeling>








# 'delta_first_reg', 'interaction_regreister_delta', 'delta_first_start',
#        'Camp_Length'

'''
    camp_list = [df6578, df6532, df6543, df6580, df6570, df6542, df6571, 
    df6527, df6526, df6539, df6528, df6555, df6541, df6523, df6538, 
    df6549, df6586, df6554, df6529, df6540, df6534, df6535, df6561, 
    df6585, df6536, df6562, df6537, df6581, df6524, df6587, df6557, 
    df6546, df6569, df6564, df6575, df6552, df6558, df6530, df6560, 
    df6531, df6544, df6565, df6553, df6563]

'''