import pandas as pd 
import numpy as np 
pd.set_option('display.max_columns', None) 
train = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Train.csv')
test = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/test.csv')
attends_df = pd.read_csv('dec21.csv')
patient_df = pd.read_csv('patient_dec24.csv')
camp_info = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Health_Camp_Detail.csv')

def impute_camp_info(dataframe):
    '''
    Map categorical information from Health_Camp to it's ID
    '''
    camp_info.Category1 = camp_info.Category1.astype(str)
    camp_info.Category2 = camp_info.Category2.astype(str)
    camp_info.Category3 = camp_info.Category3.astype(str)

    camp_ID = camp_info['Health_Camp_ID'].values
    cat1_vals = camp_info['Category1'].values
    cat2_vals = camp_info['Category2'].values
    cat3_vals = camp_info['Category3'].values
    cat3_vals2 = [x+'100' for x in cat3_vals]

    cat1d, cat2d,cat3d = list(zip(camp_ID, cat1_vals)) , list(zip(camp_ID, cat2_vals)), list(zip(camp_ID, cat3_vals2))
    cd1,cd2,cd3 = {k:v for (k,v) in cat1d} , {k:v for (k,v) in cat2d} , {k:v for (k,v) in cat3d}
    
    dataframe['Category1'] = dataframe['Health_Camp_ID'].map(cd1)
    dataframe['Category2'] = dataframe['Health_Camp_ID'].map(cd2)
    dataframe['Category3'] = dataframe['Health_Camp_ID'].map(cd3)
    
    return dataframe

def impute_missing_dates(dataframe):
    ''' 
    Merge Camp_Info with train and test 
    '''
    
    camp_info['Camp_Start_Date'] = pd.to_datetime(camp_info['Camp_Start_Date'])
    camp_info['Camp_End_Date'] = pd.to_datetime(camp_info['Camp_End_Date'])

    cci = camp_info['Health_Camp_ID'].values
    cco = camp_info['Camp_Start_Date'].values 
    ccc = camp_info['Camp_End_Date'].values 

    bla , blah2 = list(zip(cci,cco)) ,list(zip(cci,ccc))
    dict_of_dates , dict_of_dates2 = {k:v for (k,v) in bla} , {k:v for (k,v) in blah2}

    dataframe['Camp_Start_Date2'] = dataframe['Health_Camp_ID'].map(dict_of_dates)
    dataframe['Camp_End_Date2'] =  dataframe['Health_Camp_ID'].map(dict_of_dates2)

    return  dataframe

def combine(x):
    '''
    Create a column for each patient & event
    '''
    #train['patient_event'] = list(zip(train.Patient_ID,train.Health_Camp_ID))
    x.Patient_ID = x.Patient_ID.astype(int)
    x.Health_Camp_ID = x.Health_Camp_ID.astype(int)

    x.Patient_ID = x.Patient_ID.astype(str)
    x.Health_Camp_ID = x.Health_Camp_ID.astype(str)

    x['patient_event'] =  x['Patient_ID'] + x['Health_Camp_ID'] 
    x.patient_event = x.patient_event.astype(int)
 
    return x ##  

def patient_merging(dataframe):
    '''
    Merge train and Patient_info on Patient_ID
    '''
    dataframe_copy = dataframe.copy() 
    patient_df_  = patient_df.copy()  
    dataframe_copy.Patient_ID = dataframe_copy.Patient_ID.astype(int)
    dataframe_copied = pd.merge(dataframe_copy,patient_df_ , how='outer', on ='Patient_ID')
    return dataframe_copied

def to_date(dataframe):
    '''
    Convert date columns to date_time & create Length of event feature 
    '''
    dataframe['Registration_Date'].fillna('10-may-93', inplace=True) 
    # dataframe['Registration_Date'] = pd.to_datetime(dataframe['Registration_Date'], format="%d-%b-%y")
    return dataframe

def merger(dataframe):
    '''
    merge train and attendance  
    edit_y values
    ''' 'Unnamed: 0', 'Unnamed: 0.1', 'Patient_ID', 'Online_Follower'
    attends_df_ = attends_df.copy() 
    attends_df_= attends_df_.drop([
       'LinkedIn_Shared', 'Twitter_Shared', 'Facebook_Shared', 'Income',
       'Education_Score', 'Age', 'First_Interaction', 'City_Type', 'Camp_Start_Date',
       'Employer_Category', 'Job_Type',  'online_score', 'Camp_End_Date',
       'Donation', 'Health_Score', 'Unnamed: 4', 'Health_Camp_ID'], axis=1) 
    dataframe['patient_event']= dataframe.patient_event.fillna(0)
    dataframe.patient_event = dataframe.patient_event.astype(int)
    attends_df_.patient_event = attends_df_.patient_event.astype(int)

    dataframe = pd.merge(dataframe, attends_df_ ,how='outer', on='patient_event')
    dataframe['y_target'] = dataframe['y_target'].replace(to_replace = 'None', value=np.nan).fillna(0)
    x_ = dataframe[dataframe['patient_event'].notna()]
    return x_

def drop_cols(dataframe):
    '''
    dropping columns for test file
    -Adding time features 
    '''
    cols_2_drop = [
    'LinkedIn_Shared','Twitter_Shared', 'Facebook_Shared', 'Income', 'Education_Score', 'Age']
    dataframe = dataframe.drop(cols_2_drop, axis=1)

    #x['Camp_Start_Date2'] = pd.to_datetime(x['Camp_Start_Date2'], format="%d-%m-%y")
    dataframe['Registration_Date'] = pd.to_datetime(dataframe['Registration_Date'])
    dataframe['First_Interaction'] = pd.to_datetime(dataframe['First_Interaction'])

    dataframe['delta_first_reg'] = dataframe['Camp_Start_Date2'] - dataframe['Registration_Date'] #Check start - regrister
    dataframe['delta_first_reg'] = dataframe['delta_first_reg'].dt.days

    dataframe['interaction_regreister_delta'] = dataframe['Registration_Date'] - dataframe['First_Interaction'] #check regrister - interaction 
    dataframe['delta_first_start'] = dataframe['Camp_Start_Date2'] - dataframe['First_Interaction'] # Check  startdate - first interaction 
    
    dataframe['delta_reg_end'] = dataframe['Camp_End_Date2'] - dataframe['Registration_Date']
    return dataframe 

def impute_missing_vals(dataframe):
    '''
    For Category1 
    y_target.fillna(), fill rest of dates with subtraction 
    - removing string part , keeping only int
    '''
    dataframe['Camp_Length'] =  dataframe['Camp_End_Date2'] - dataframe['Camp_Start_Date2']
    
    return dataframe

def keep_ints(dataframe):
    '''
    Start Conversion for Data_Time features into usable object:
    ['delta_first_start','interaction_regreister_delta','Camp_Length']
    '''
    dataframe.delta_first_start = dataframe.delta_first_start.astype(str)
    xi = dataframe.delta_first_start.values
    xi_ = [] 

    for i in xi:
        xi_.append(''.join([x for x in i if x in '-1234567890']) )

    dataframe.interaction_regreister_delta = dataframe.interaction_regreister_delta.astype(str)
    xii = dataframe.interaction_regreister_delta.values
    xii_ = [] 

    for i in xii:
        xii_.append(''.join([x for x in i if x in '-1234567890']) )

    dataframe.Camp_Length = dataframe.Camp_Length.astype(str)
    xiii = dataframe.Camp_Length.values
    xiii_ = [] 

    for i in xiii:
        xiii_.append(''.join([x for x in i if x in '-1234567890']) )

    dataframe.delta_reg_end = dataframe.delta_reg_end.astype(str)
    xiiii = dataframe.delta_reg_end.values
    xiiii_ = [] 

    for i in xiiii:
        xiiii_.append(''.join([x for x in i if x in '-1234567890']) )


    dataframe['Camp_Length'] = xiii_
    dataframe['interaction_regreister_delta'] = xii_
    dataframe['delta_first_start'] = xi_ 
    dataframe['delta_reg_end'] = xiiii_
    return dataframe 


if __name__ == '__main__':
    train, test = impute_camp_info(train), impute_camp_info(test)
    #print(train.info() , test.info())
    train , test = impute_missing_dates(train) , impute_missing_dates(test)
    train_event_patID ,  test_event_patID = combine(train) , combine(test)
    train_concat , test_concat = patient_merging(train_event_patID) , patient_merging(test_event_patID) 
    train_dated , test_dated = to_date(train_concat) , to_date(test_concat)

    train_merge ,test_merge = merger(train_dated), merger(test_dated)
    droped_train, droped_test = drop_cols(train_merge), drop_cols(test_merge)
    
    train_final, test_final = impute_missing_vals(droped_train), impute_missing_vals(droped_test)
    train_final2, test_final2 = keep_ints(train_final) , keep_ints(test_final)
    
    checker = train_final2[train_final2['Health_Camp_ID'].notnull()]
    print(checker.info(), checker.describe())
    #checker_  = convert_date_part2(checker) #, convert_date_part2(test_final2)
    #checker.to_csv('ready12_24_train.csv', index = False)











    # attends_df.patient_event = attends_df.patient_event.astype(int)
    # train_dated.patient_event = train_dated.patient_event.astype(int)
    # test_dated.patient_event = test_dated.patient_event.astype(int)


    # dict_ = {}
    # for i in train_concat['patient_event'].values:
    #     i_ = str(i)
    #     if i_ not in dict_:
    #         dict_[i_]=1
    #     else:
    #         dict_[i_]+=1
    # check = [(x,xx) for (x,xx) in dict_.items() if xx>1]
    #print(check) 

    #print(train_concat[ (train_concat['Health_Camp_ID_y'] == 6578.0) | (train_concat['Health_Camp_ID_x'] == 6560.0) ]) 

# Tasks for this file 
# combine df with train_df ,  combine df with test_df 





'''
def patient_merging():
  #
    Merge train and test with patient_df ON patient_event column
    remove all values not in both train/test AND patient_ID
    #
    attends_df_ = attends_df.copy()
    attends_df_.patient_event = attends_df_.patient_event.astype(str)
    train_copy = train_dated.copy()
 
    train_merged = pd.merge(attends_df_,train_copy, how='outer', on ='patient_event')
    #train_merged2 = pd.merge(df_2, train_merged, how='outer', on =['Patient_ID'] )

    return train_merged

changed after 13:33 on 12/22 

'''


# def sanity():
#     '''
#     Checking for overlap and sanity
#     overlap among patient_event for

#     train and test == SHOULD BE ZERO
#     train and attends_df should NOT be ZERO
#     test and attends_df should NOT be ZERO 
#     '''
#     attends = set(attends_df['patient_event'].values) 
#     trainss = set(train_dated['patient_event'].values)
#     testss = set(test_dated['patient_event'].values)

#     at = set(attends_df['Patient_ID'].values)
#     tts = set(train_dated['Patient_ID'].values)
#     tes = set(test_dated['Patient_ID'].values)

#     att_event = set(attends_df['Health_Camp_ID'].values)
#     train_event = set(train_dated['Health_Camp_ID'].values)
#     test_event = set(test_dated['Health_Camp_ID'].values) 

#     overlap1 = att_event.difference(test_event) # no overlap
#     overlap2 = att_event.difference(train_event) # no overlap
#     overlap3 = test_event.difference(train_event  ) # no overlap
#     overlap4 = trainss.intersection(attends) # no overlap ?
#     overlap5 = trainss.union(testss) # should be size of both 
#     overlap6 = trainss.difference(testss) # ~ 75278 

#     return  [len(overlap1),len(overlap2), len(overlap3)]


# def explore():
#     ''' self explore EDA '''
#     tst = test_merge[test_merge['y_target'] == 1 ]
#     trn = train_merge[train_merge['y_target'] == 1] 
#     tst_checker = pd.merge(tst,trn, how='outer', on = 'patient_event')
#     print(type(tst_checker))
#     return tst_checker 
# first glance at 12/23/ 12am
# need to map some dicts , start_date, 
