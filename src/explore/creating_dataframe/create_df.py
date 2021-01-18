import pandas as pd 
import numpy as np 
pd.set_option('display.max_columns', None) 

patient = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/patient_attendance.csv') 
event1 = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/First_Health_Camp_Attended.csv')
event2 = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Second_Health_Camp_Attended.csv')
event3 = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Third_Health_Camp_Attended.csv')
camp_info = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Health_Camp_Detail.csv')
df_city = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/D7.csv') 

def impute_city(patient):
    '''
    Edit column for City_Type, impute missing values
    '''
    dict_of_cities = {}
    for i in df_city['City_Type'].values:
        if i not in dict_of_cities:
            dict_of_cities[i]=1
        else:
            dict_of_cities[i]+=1

    patient['City_Type2'] = df_city['City_Type'].map(dict_of_cities)
    return patient 

def impute_online_score(patient):
    '''
    Create Column for sum of all Online Shares 
    '''
    #sum col values in certain columns across each individual row
    online_score = patient['Online_Follower'] + patient['LinkedIn_Shared'] + patient['Facebook_Shared'] + patient['Twitter_Shared'] 
    patient['online_score'] = online_score 
    return patient
    
def make_target(event_df):
    ''' # might need to redo this since there will be events in Train / Test where a patient did not attend, 
    given that people might have went to event two
    Create Target Value 
    '''
    if event_df.shape == event1.shape:
        event_df['y_target'] = event_df['Health_Score'].apply(lambda x:1 if x >0 else 0)

    if event_df.shape == event2.shape:
        event_df['y_target'] = event_df['Health Score'].apply(lambda x:1 if x >0 else 0)

    if event_df.shape == event3.shape:
        event_df['y_target'] = event_df['Number_of_stall_visited'].apply(lambda x:1 if x>0 else 0)

    return event_df  

def make_combo(dataframe):
    '''
    Create a column for each patient & event
    '''
    dataframe.Patient_ID = dataframe.Patient_ID.astype(int)
    dataframe.Health_Camp_ID = dataframe.Health_Camp_ID.astype(int)

    dataframe.Patient_ID = dataframe.Patient_ID.astype(str)
    dataframe.Health_Camp_ID = dataframe.Health_Camp_ID.astype(str)
    dataframe['patient_event'] = dataframe['Patient_ID'] + dataframe['Health_Camp_ID']

    return dataframe #dataframe ## this is run 3 times, once for each of the 3 CSVs


def combine_info(dataframe):
    '''
    Merge patient info with Health_Camp_Detail
    '''
    patient_copy = patient.copy() 
    patient_copy.Patient_ID = patient_copy.Patient_ID.astype(str)
    combined_df = pd.merge(patient_copy,dataframe, on=['Patient_ID']) # this makes adds the patient info columns
                                                                        # to each ->  camp1,2,3
    #print(combined_df.shape, 'first combinefirst combinefirst combinefirst combinefirst combine')
   
    return combined_df

def combine_camp_detail(dataframe):
    '''
    Merge Health_Camp_Detail with merged Patient_Event
    '''
    camp_info_copy = camp_info.copy()
    camp_info_copy.Health_Camp_ID = camp_info_copy.Health_Camp_ID.astype(str)
    combined_df = pd.merge(camp_info_copy,dataframe, on=['Health_Camp_ID'])
    #print(combined_df, 'combined df.shape') ## this is run 3 times to merge camp info and align date columns for each of the 3 CSVs
    return combined_df 

def all_camps(x,y,z):
    '''
    concat all instances of a patient attending a camp
    '''
    concat1 = pd.concat([x,y,z])
    return concat1

def to_date(concat1): 
    '''
    Convert date columns to date_time & create Length of event feature 
    These will remain in the concatted successful attendance 
    '''
    concat1['Camp_Start_Date'].fillna('10-may-93', inplace=True) 
    concat1['Camp_End_Date'].fillna('10-may-93', inplace=True)
    concat1['First_Interaction'].fillna('10-may-93', inplace=True) 

    concat1['Camp_Start_Date'] = pd.to_datetime(concat1['Camp_Start_Date'], format="%d-%b-%y") 
    concat1['Camp_End_Date'] = pd.to_datetime(concat1['Camp_End_Date'], format="%d-%b-%y")
    concat1['First_Interaction'] = pd.to_datetime(concat1['First_Interaction'], format="%d-%b-%y")

    concat1['Camp_length'] = concat1['Camp_End_Date'] - concat1['Camp_Start_Date']
    #print(concat1.shape , 'main_df.shape ********************************************')
    return concat1  

def to_date_patient(patient): # might not need since its already attached to each patient in the concat1 df
    '''   # And it changes
    Convert date column in patient DF
    '''
    patient['First_Interaction'] = pd.to_datetime(patient['First_Interaction'], format="%d-%b-%y")
    return patient 

if __name__ == '__main__':
    impute_citi = impute_city( patient )
    impute_online = impute_online_score(impute_citi)
    print(impute_citi.info(), impute_citi.head(10))
    #patient2 = make_target_patient(patient)
    event1,event2,event3 = make_target(event1) , make_target(event2) , make_target(event3)

    event1_, event2_,event3_= make_combo(dataframe=event1) , make_combo(dataframe=event2) , make_combo(dataframe=event3)
    event1_a, event2_a,event3_a= combine_info(dataframe=event1_) , combine_info(dataframe=event2_) , combine_info(dataframe=event3_)
    event1_b, event2_b,event3_b=combine_camp_detail(dataframe=event1_a), combine_camp_detail(dataframe=event2_a), combine_camp_detail(dataframe=event3_a)
    concat1 = all_camps(event1_b, event2_b , event3_b)

 
    dated = to_date(concat1) # dates for all 20,555 patient_attends 
    dated_patient = to_date_patient(impute_online ) 
    d1,d2,d3 = {},{},{}
    for i in dated['Age'].values:
        if i not in d1:
            d1[i]=1
        else:
            d1[i]+=1
    for i in dated['Education_Score'].values:
        if i not in d2:
            d2[i]=1
        else:
            d2[i]+=1    
    for i in dated['Income'].values:
        if i not in d3:
            d3[i] =1
        else:
            d3[i]+=1
    print(d1,d2,d3)

    age_and_edu = dated[(dated['Age'] ==0) & (dated['Education_Score']==0)]
    print(len(age_and_edu))



    # dated.to_csv('dec21.csv', index=False) # as at 10am 12/22 dec21.csv is all patients
    #dated_patient.to_csv('patient_dec24.csv', index=False)

 