from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#knn_plotting.py  import plot_distances
plt.rcParams['figure.dpi'] = 200 
train_set = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/train_1.csv')
test_set = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/test_df1.csv') 

train1 = train_set.drop(['Unnamed: 0.1.1','Unnamed: 0', 'Unnamed: 0.1', 'Registration_Date','Employer_Category','Patient_ID','First_Interaction', 'Health_Camp_ID','Income',
'Education_Score','Age', 'City_Type', 'Job_Type', 'Education_Score2',
'Education_Scorez',],axis=1) 

test_set = test_set.drop(['Unnamed: 0.1.1','Unnamed: 0', 'Unnamed: 0.1','Registration_Date','Employer_Category','Patient_ID','First_Interaction', 'Health_Camp_ID','Income',
'Education_Score','Age', 'City_Type', 'Job_Type', 'Education_Score2',
'Education_Scorez',],axis=1) 

y1, y2 = train1.pop('Event1_or_2') , test_set.pop('Event1_or_2')
print(test_set.columns) 