import pandas as pd 
import numpy as np 
pd.set_option('display.max_columns', None) 





class Create_DF:

    def __init__(self ): #name
        #self.name = name 
        self.patient = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Patient_Profile.csv') 
        self.event1 = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/First_Health_Camp_Attended.csv')
        self.event2 = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Second_Health_Camp_Attended.csv')
        self.event3 = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Third_Health_Camp_Attended.csv')
        self.camp_info = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Health_Camp_Detail.csv')
        self.df_city = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/D7.csv') 
        self.dict_of_cities = {} 

    def impute_city(self):
        '''
        Edit column for City_Type, impute missing values
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
        Edit column for City_Type, impute missing values
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
        Create Column for sum of all Online Shares 
        '''
        #sum col values in certain columns across each individual row
        self.online_score = self.patient['LinkedIn_Shared'] + self.patient['Facebook_Shared'] + self.patient['Twitter_Shared'] + self.patient['Online_Follower'] 
        self.patient['online_score'] = self.online_score 
        return self.patient

    def make_target(self):
        ''' 

        '''
        
        self.event1['y_target'] = self.event1['Health_Score'].apply(lambda x:1 if x >0 else 0)
        self.event2['y_target'] = self.event2['Health Score'].apply(lambda x:1 if x >0 else 0)
        self.event3['y_target'] = self.event3['Number_of_stall_visited'].apply(lambda x:1 if x>0 else 0)

        return self.event1 ,self.event2,self.event3

    def make_primary_key(self):
        '''
        *Better way to do this ?!? 
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
        Merge patient info with Health_Camp_Detail
        '''
        self.patient_copy = self.patient.copy() 
        self.patient_copy.Patient_ID = self.patient_copy.Patient_ID.astype(str)
        self.combined_df1 = pd.merge(self.patient_copy,self.event1, on=['Patient_ID']) # this makes adds the patient info columns
        self.combined_df2 = pd.merge(self.patient_copy,self.event2, on=['Patient_ID'])                                                     
        self.combined_df3 = pd.merge(self.patient_copy,self.event3, on=['Patient_ID'])   
    
        return self.combined_df1, self.combined_df2 , self.combined_df3 

    def merge_patient_camps(self):
        '''
        
        '''
        self.camp_info_copy = self.camp_info.copy() 
        self.camp_info_copy.Health_Camp_ID = self.camp_info_copy.Health_Camp_ID.astype(str)

        self.combined_df1 = pd.merge(self.camp_info_copy, self.combined_df1, on=['Health_Camp_ID'] )
        self.combined_df2 = pd.merge(self.camp_info_copy, self.combined_df2, on=['Health_Camp_ID'])
        self.combined_df3 = pd.merge(self.camp_info_copy, self.combined_df3, on=['Health_Camp_ID'] )

        return self.combined_df1, self.combined_df2 , self.combined_df3 

    def combine_all_camps(self):
        '''
        merge all 3 camps into single
        '''
        self.all_camps = pd.concat([self.combined_df1,self.combined_df2,self.combined_df3])
        return self.all_camps

    def impute_dates(self):
        '''
        Convert date columns to date_time & create Length of event feature 
        '''
        # print(self.all_camps['Camp_Start_Date'].values)
        # self.all_camps['Camp_Start_Date'] = self.all_camps['Camp_Start_Date'].fillna('10-may-93', inplace=True)
        # self.all_camps['Camp_End_Date'] = self.all_camps['Camp_End_Date'].fillna('10-may-93', inplace=True)
        # self.all_camps['First_Interaction'] = self.all_camps['First_Interaction'].fillna('10-may-93', inplace=True)
         
        self.all_camps['Camp_Start_Date'] = pd.to_datetime(self.all_camps['Camp_Start_Date'], format="%d-%b-%y") 
        self.all_camps['Camp_End_Date'] = pd.to_datetime(self.all_camps['Camp_End_Date'], format="%d-%b-%y")
        self.all_camps['First_Interaction'] = pd.to_datetime(self.all_camps['First_Interaction'], format="%d-%b-%y")

        self.all_camps['Camp_length'] = self.all_camps['Camp_End_Date'] - self.all_camps['Camp_Start_Date']
        
        return self.all_camps

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
    df_check = impute_citi.impute_dates()
    print(df_check.info())


'''
8/11/21
-Need to disentangle the two main data frames(csv files) being worked on
'''