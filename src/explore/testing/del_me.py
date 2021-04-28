import pandas as pd 
import numpy as np 
pd.set_option('display.max_columns', None) 
 
from preprocessing import drop_cols , one_hot_encoding , scale
from city_testing import run_test_typeAA , run_test_typeC
import itertools as it 
df_no_scale = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/temp_csv/hot_code_NO_scale.csv')
ids = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Train.csv')
#del df_no_scale['Health_Camp_ID']
del df_no_scale['Category1_x']
del df_no_scale['Category2']
del df_no_scale['Category3']
del df_no_scale['City_Type']
del df_no_scale['Job_Type']
del df_no_scale['online_score']
 
df_no_scale['Patient_ID'] = ids['Patient_ID'].values


df_no_scale.to_csv('/home/allen/Galva/capstones/capstone2/src/explore/testing/hoc.csv', index=False)