import pandas as pd 
import numpy as np 
pd.set_option('display.max_columns', None) 

patient = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Patient_Profile.csv') 
event1 = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/First_Health_Camp_Attended.csv')
event2 = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Second_Health_Camp_Attended.csv')
event3 = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Third_Health_Camp_Attended.csv')
camp_info = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Health_Camp_Detail.csv')
df_city = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/D7.csv') 



class Create_DF:

    def __init__(self,name,patient, df_city):
        self.name = name 
        self.patient = patient  
        self.df_city = df_city 


    def impute_city(self,patient, df_city):
        '''
        Edit column for City_Type, impute missing values
        '''
        self.dict_of_cities = {} 
        for i in self.df_city['City_Type'].values:
            if i not in self.dict_of_cities:
                self.dict_of_cities[i]=1
            else:
                self.dict_of_cities[i]+=1

        self.patient['City_Type2'] = self.df_city['City_Type'].map(self.dict_of_cities)
        return self.patient 

    def impute_Job(self,patient):
        '''
        Edit column for City_Type, impute missing values
        '''
        self.patient['Employer_Category'] = self.patient['Employer_Category'].astype(str)
        
        self.patient['Employer_Category'] = patient['Employer_Category'].replace(to_replace = 'None', value=np.nan).fillna(0)
        self.to_change = self.patient['Employer_Category'].values 
        self.to_change_ = []
        for i in self.to_change:
            if i != 'nan':
                self.to_change_.append(i)
            else:
                self.to_change_.append('9999') 

        
        self.patient['Job Type'] = self.to_change_
        print(patient.info())
 
        return self.patient  

    def impute_online_score(self,patient):
        '''
        Create Column for sum of all Online Shares 
        '''
        #sum col values in certain columns across each individual row
        self.online_score = self.patient['LinkedIn_Shared'] + self.patient['Facebook_Shared'] + self.patient['Twitter_Shared'] + self.patient['Online_Follower'] 
        self.patient['online_score'] = self.online_score 
        return self.patient



if __name__ == '__main__':
    impute_citi = Create_DF(patient, camp_info, df_city  )
    impute_citi.impute_city(patient, df_city)
    impute_citi.impute_Job(patient)
    df_check = impute_citi.impute_online_score(patient )
    print(patient.head())
      
    print(df_check)

'''
5/12 Got first two functions to work - can get a data frame
-not 
'''