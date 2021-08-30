'''
This the Object oriented programming code that wrangles the files into the testing format
'''



import pandas as pd 
import numpy as np 
pd.set_option('display.max_columns', None) 

class Create_DF:

    def __init__(self ): 
        
        self.patient = pd.read_csv('/home/allen/RIP_Tensor1/capstone2/Capstone2/data/Train/Patient_Profile.csv') 
        self.event1 = pd.read_csv('/home/allen/RIP_Tensor1/capstone2/Capstone2/data/Train/First_Health_Camp_Attended.csv')
        self.event2 = pd.read_csv('/home/allen/RIP_Tensor1/capstone2/Capstone2/data/Train/Second_Health_Camp_Attended.csv')
        self.event3 = pd.read_csv('/home/allen/RIP_Tensor1/capstone2/Capstone2/data/Train/Third_Health_Camp_Attended.csv')
        self.camp_info = pd.read_csv('/home/allen/RIP_Tensor1/capstone2/Capstone2/data/Health_Camp_Detail.csv')
        self.df_city = pd.read_csv('/home/allen/RIP_Tensor1/capstone2/Capstone2/data/D7.csv') 
        self.dict_of_cities = {} 
        self.train = pd.read_csv('/home/allen/RIP_Tensor1/capstone2/Capstone2/data/Train/Train.csv')
        self.test = pd.read_csv('/home/allen/RIP_Tensor1/capstone2/Capstone2/data/Train/test.csv')
        self.attends_df = pd.read_csv('/home/allen/RIP_Tensor1/capstone2/Capstone2/data/attends_df.csv')




    def impute_city(self):
        '''
        Input: patient.csv (patient_DataFrame), camp & location dictionary (D7.csv)
        Action: imputes missing location values 
        Output: patient_DataFrame 
        '''
        
        for i in self.df_city['City_Type'].values:
            if i not in self.dict_of_cities:
                self.dict_of_cities[i]=1
            else:
                self.dict_of_cities[i]+=1

        self.patient['City_Type2'] = self.df_city['City_Type'].map(self.dict_of_cities)
        return self.patient 

    def impute_Job(self ):
        '''
        Input: patient_DataFrame
        Action: imputes the value '9999' for 'Employer_Category'
        Output: patient_DataFrame
        
        '''
        self.patient['Employer_Category'] = self.patient['Employer_Category'].astype(str)
        
        self.patient['Employer_Category'] = self.patient['Employer_Category'].replace(to_replace = 'None', value=np.nan).fillna(0)
        self.to_change = self.patient['Employer_Category'].values 
        self.to_change_ = []
        for i in self.to_change:
            if i != 'nan':
                self.to_change_.append(i)
            else:
                self.to_change_.append('9999') 

         
        self.patient['Job Type'] = self.to_change_
         
        return self.patient  

    def impute_online_score(self ):
        '''
        Input: patient_DataFrame
        Action: Create a new column 'Online_score' for each patient 
        Output: patient_DataFrame
         
        '''
        
        self.online_score = self.patient['LinkedIn_Shared'] + self.patient['Facebook_Shared'] + self.patient['Twitter_Shared'] + self.patient['Online_Follower'] 
        self.patient['online_score'] = self.online_score 
        return self.patient

    

    def make_target(self):
        ''' 
        Input: event1,event2,event3 DataFrames
        Action: Impute y_target for model prediction based on patient attendance 
        Output: event DataFrames
        '''
        
        self.event1['y_target'] = self.event1['Health_Score'].apply(lambda x:1 if x >0 else 0)
        self.event2['y_target'] = self.event2['Health Score'].apply(lambda x:1 if x >0 else 0)
        self.event3['y_target'] = self.event3['Number_of_stall_visited'].apply(lambda x:1 if x>0 else 0)

        return self.event1 ,self.event2,self.event3

    def make_primary_key(self):
        '''
        Input: event1,event2,event3 DataFrames
        Action: create a primary key from patient_event 
        Output: event DataFrames
        '''
    
        self.event1.Patient_ID = self.event1.Patient_ID.astype(int)
        self.event1.Health_Camp_ID = self.event1.Health_Camp_ID.astype(int)

        self.event1.Patient_ID = self.event1.Patient_ID.astype(str)
        self.event1.Health_Camp_ID = self.event1.Health_Camp_ID.astype(str)

        self.event2.Patient_ID = self.event2.Patient_ID.astype(int)
        self.event2.Health_Camp_ID = self.event2.Health_Camp_ID.astype(int)

        self.event2.Patient_ID = self.event2.Patient_ID.astype(str)
        self.event2.Health_Camp_ID = self.event2.Health_Camp_ID.astype(str)

        self.event3.Patient_ID = self.event3.Patient_ID.astype(int)
        self.event3.Health_Camp_ID = self.event3.Health_Camp_ID.astype(int)

        self.event3.Patient_ID = self.event3.Patient_ID.astype(str)
        self.event3.Health_Camp_ID = self.event3.Health_Camp_ID.astype(str)


        self.event1['patient_event'] = self.event1['Patient_ID'] + self.event1['Health_Camp_ID']
        self.event2['patient_event'] = self.event2['Patient_ID'] + self.event2['Health_Camp_ID']
        self.event3['patient_event'] = self.event3['Patient_ID'] + self.event3['Health_Camp_ID']

        return self.event1,self.event2,self.event3

    def combine_info(self):
        '''
        Input: event1,event2,event3, patient_DataFrame  
        Action: Merge patient_DataFrame with health camp events  
        Output: event DataFrames with patient_info columns 
    
        '''
        self.patient_copy = self.patient.copy() 
        self.patient_copy.Patient_ID = self.patient_copy.Patient_ID.astype(str)
        self.combined_df1 = pd.merge(self.patient_copy,self.event1, on=['Patient_ID']) # this makes adds the patient info columns
        self.combined_df2 = pd.merge(self.patient_copy,self.event2, on=['Patient_ID'])                                                     
        self.combined_df3 = pd.merge(self.patient_copy,self.event3, on=['Patient_ID'])   
    
        return self.combined_df1, self.combined_df2 , self.combined_df3 

    def merge_patient_camps(self):
        '''
        Input:  event_dataFrame, camp_information_DataFrame
        Action: Use primary key to merge camp_information & event_DataFrame
        Output: Data Frame with all camp_info, patient_info, & health camp specifics 
        '''
        self.camp_info_copy = self.camp_info.copy() 
        self.camp_info_copy.Health_Camp_ID = self.camp_info_copy.Health_Camp_ID.astype(str)

        self.combined_df1 = pd.merge(self.camp_info_copy, self.combined_df1, on=['Health_Camp_ID'] )
        self.combined_df2 = pd.merge(self.camp_info_copy, self.combined_df2, on=['Health_Camp_ID'])
        self.combined_df3 = pd.merge(self.camp_info_copy, self.combined_df3, on=['Health_Camp_ID'] )

        return self.combined_df1, self.combined_df2 , self.combined_df3 

    def combine_all_camps(self):
        '''
        Input: The combo dataframes
        Action: Concat all 3 camps DataFrames into single DataFrame 
        Output: Concatted DataFrame        
        '''
        self.all_camps = pd.concat([self.combined_df1,self.combined_df2,self.combined_df3])

        return self.all_camps

    def impute_dates(self):
        '''
        Input:
        Action:
        Output:
        Convert date columns to date_time & create Length of event feature 
        '''
         
        self.all_camps['Camp_Start_Date'] = pd.to_datetime(self.all_camps['Camp_Start_Date'], format="%d-%b-%y") 
        self.all_camps['Camp_End_Date'] = pd.to_datetime(self.all_camps['Camp_End_Date'], format="%d-%b-%y")
        self.all_camps['First_Interaction'] = pd.to_datetime(self.all_camps['First_Interaction'], format="%d-%b-%y")

        self.all_camps['Camp_length'] = self.all_camps['Camp_End_Date'] - self.all_camps['Camp_Start_Date']
        print(self.all_camps.shape)
        return self.all_camps

    def to_date_patient(self):  
        '''    
        Input:
        Action:
        Output:
        Convert date column in patient DF
        '''
        self.patient['First_Interaction'] = pd.to_datetime(self.patient['First_Interaction'], format="%d-%b-%y")
        return self.patient 

    def impute_missing_camp_info(self):
        '''
        Input:
        Action:
        Output:
        Map categorical information from Health_Camp file to Main df by Camp_ID
        '''
        self.camp_info.Category1 = self.camp_info.Category1.astype(str)
        self.camp_info.Category2 = self.camp_info.Category2.astype(str)
        self.camp_info.Category3 = self.camp_info.Category3.astype(str)

        self.camp_ID = self.camp_info['Health_Camp_ID'].values
        self.cat1_vals = self.camp_info['Category1'].values
        self.cat2_vals = self.camp_info['Category2'].values
        self.cat3_vals = self.camp_info['Category3'].values
        self.cat3_vals2 = [x+'100' for x in self.cat3_vals]

        self.cat1d = list(zip(self.camp_ID, self.cat1_vals))  
        self.cat2d = list(zip(self.camp_ID, self.cat2_vals)) 
        self.cat3d = list(zip(self.camp_ID, self.cat3_vals2))

        self.cd1 = {k:v for (k,v) in self.cat1d}
        self.cd2 = {k:v for (k,v) in self.cat2d}
        self.cd3 = {k:v for (k,v) in self.cat3d}

        self.train['Category1'] = self.train['Health_Camp_ID'].map(self.cd1)
        self.train['Category2'] = self.train['Health_Camp_ID'].map(self.cd2)
        self.train['Category3'] = self.train['Health_Camp_ID'].map(self.cd3)

        self.test['Category1'] = self.test['Health_Camp_ID'].map(self.cd1)
        self.test['Category2'] = self.test['Health_Camp_ID'].map(self.cd2)
        self.test['Category3'] = self.test['Health_Camp_ID'].map(self.cd3)

        return self.train , self.test

    def impute_missing_dates(self):
        ''' 
        Input:
        Action:
        Output:
        Merge Camp_Info features with train/test dataframes via mapping
        '''
        
        self.camp_info['Camp_Start_Date'] = pd.to_datetime(self.camp_info['Camp_Start_Date'])
        self.camp_info['Camp_End_Date'] = pd.to_datetime(self.camp_info['Camp_End_Date'])

        self.cci = self.camp_info['Health_Camp_ID'].values
        self.cco = self.camp_info['Camp_Start_Date'].values 
        self.ccc = self.camp_info['Camp_End_Date'].values 

        self.bla , self.blah2 = list(zip(self.cci,self.cco)) ,list(zip(self.cci,self.ccc))
        self.dict_of_dates , self.dict_of_dates2 = {k:v for (k,v) in self.bla} , {k:v for (k,v) in self.blah2}

        self.train['Camp_Start_Date2'] = self.train['Health_Camp_ID'].map(self.dict_of_dates)
        self.train['Camp_End_Date2'] =  self.train['Health_Camp_ID'].map(self.dict_of_dates2)

        self.test['Camp_Start_Date2'] = self.test['Health_Camp_ID'].map(self.dict_of_dates)
        self.test['Camp_End_Date2'] =  self.test['Health_Camp_ID'].map(self.dict_of_dates2)

        return  self.train , self.test


    def create_primary_key(self):
        '''
        Input: 
        Action:
        Output:
        Create a primary key for each patient from their ID and Camp_Info
        '''
        self.train.Patient_ID = self.train.Patient_ID.astype(int)
        self.train.Health_Camp_ID = self.train.Health_Camp_ID.astype(int)

        self.test.Patient_ID = self.test.Patient_ID.astype(int)
        self.test.Health_Camp_ID = self.test.Health_Camp_ID.astype(int)

        self.train.Patient_ID = self.train.Patient_ID.astype(str)
        self.train.Health_Camp_ID = self.train.Health_Camp_ID.astype(str)

        self.test.Patient_ID = self.test.Patient_ID.astype(str)
        self.test.Health_Camp_ID = self.test.Health_Camp_ID.astype(str)

        self.train['patient_event'] = self.train["Patient_ID"] + self.train['Health_Camp_ID']
        self.test['patient_event'] = self.test["Patient_ID"] + self.test['Health_Camp_ID']


        return self.train , self.test

    def patient_merging(self):
        '''
        Input: 
        Action:
        Output:
        Merge train and Patient_info on Patient_ID
        '''
        self.train_=self.train.copy()
        self.test_=self.test.copy()
        self.patient_copy2  =self.patient.copy()

        self.train_['Patient_ID'] = self.train['Patient_ID'].astype(int)
        self.test_['Patient_ID'] = self.test['Patient_ID'].astype(int) 
        
        self.test = pd.merge(self.test_, self.patient_copy2 , how='outer',on='Patient_ID')
        self.train = pd.merge(self.train_, self.patient_copy2 , how='outer',on='Patient_ID')
        return self.train , self.test 

    def to_date(self):
        '''
        Input: 
        Action:
        Output:
        
        Impute missing column values for Employer_Category
        '''
        self.train['Registration_Date'].fillna('10-may-93', inplace=True) 
        self.test['Registration_Date'].fillna('10-may-93', inplace=True) 

        return self.train , self.test

    # def merger(self):
    #     '''
    #     -Merge train and attendance dataframes
    #     -Remove overlap and NA values
    #     '''  
    #     self.attends_df_ = self.attends_df.copy() 
    #     self.attends_df_= self.attends_df_.drop([
    #     'LinkedIn_Shared', 'Twitter_Shared', 'Facebook_Shared', 'Income',
    #     'Education_Score', 'Age', 'First_Interaction', 'City_Type', 'Camp_Start_Date',
    #     'Employer_Category',   'online_score', 'Camp_End_Date',
    #     'Donation', 'Health_Score', 'Unnamed: 4', 'Health_Camp_ID'], axis=1) 
    #     dataframe['patient_event']= dataframe.patient_event.fillna(0)
    #     dataframe.patient_event = dataframe.patient_event.astype(int)
    #     attends_df_.patient_event = attends_df_.patient_event.astype(int)

    #     dataframe = pd.merge(dataframe, attends_df_ ,how='outer', on='patient_event')
    #     dataframe['y_target'] = dataframe['y_target'].replace(to_replace = 'None', value=np.nan).fillna(0)
    #     x_ = dataframe[dataframe['patient_event'].notna()]
        
    #     return x_




if __name__ == '__main__': 
     
    impute_citi = Create_DF(  ) #name
    impute_citi.impute_city( )
    impute_citi.impute_Job( )
    impute_citi.impute_online_score(  ) 
    impute_citi.make_target()
    impute_citi.make_primary_key()  
    impute_citi.combine_info()
    impute_citi.merge_patient_camps()
    impute_citi.combine_all_camps()
    impute_citi.impute_dates()
    impute_citi.to_date_patient() 
    impute_citi.impute_missing_camp_info()
    impute_citi.impute_missing_dates()
    impute_citi.create_primary_key()
    impute_citi.patient_merging()
    df_check = impute_citi.to_date()
    print(df_check[0].info())


#df_check = patient_df / patient_dec24.csv
