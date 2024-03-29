
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
 
pd.set_option('display.max_columns', None) 
plt.figure(dpi=200)
ids = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/testing/x1.csv')
xg = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/testing/x2.csv')
k2 = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/testing/K2.csv')
s2 = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/testing/S2.csv')
    
def combine(lst):

    new_df = ids.copy()

    for i,ii in enumerate(lst):
        if type(ii) != list:
            proba_=ii.Proba.values
            prediction_=ii.prediction.values

            st_proba = 'proba_' + lst[3][i]
            st_predict = 'prediction_' + lst[3][i] 
            
            new_df[st_proba] = proba_ 
            new_df[st_predict] = prediction_
        

    return new_df 


def edit(df): 
    
    new_df = df.copy()
    new_df['Y_count_allModels'] = df['prediction_kNN'] + df['prediction_sVC']+ df['prediction_xg']

    return new_df 


def conditions(df):
    
    score = df['y_target'].values
    score_=[]
    for i in score: 
        if i ==1:
            score_.append(1)
        else:
            score_.append(0)
    df['Y_target_SUM'] = df['Y_count_allModels'] + score_ 

    return df
       
    
def graph(new_df):
    to_print = pd.DataFrame({

        'knn_predict' :new_df['prediction_kNN'].values ,
        'xg_predict'  : new_df['prediction_xg'].values ,
        'src_predict ' : new_df['prediction_sVC'].values,
        'knn_proba' : new_df['proba_kNN'].values,
        'xg_proba'  : new_df['proba_xg'].values,
        'src_proba' : new_df['proba_sVC'].values
    })
    #lines = to_print.plot.line()
    #plt.hist(x=new_df['Y_count'])
    g = sns.relplot(data = new_df, x='Y_target_SUM', y ='Y_count_allModels', col='y_target', palette = 'deep', kind='scatter')
      
    plt.show() 
    return None 

    
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler =  StandardScaler()
pd.set_option('display.max_columns', None) 
df1 = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/ready12_24_train.csv') 
df=df1.copy() 

def drop_cols(df):
    '''
    Drop columns 
    '''
    # Health_Camp_ID 
    drop_thez=[ 'Camp Start Date - Registration Date', 'Registration Date - First Interaction', 
                    'Camp Start Date - First Interaction', 'Camp End Date - Registration Date']

    df_ = df.drop(drop_thez, axis =1)
    return df_

def drop_cols_specific(df):
    '''
    Drop columns: edit for dropping features and testing for improvement - personal grid searching 
    '''
    #Health_Camp_ID 
    drop_thez=[ 'Patient_ID_x', 'Registration_Date', 'Category1_y','Camp_Start_Date2', 'Camp_length',
    'Camp_End_Date2', 'patient_event',  'Online_Follower_x', 'First_Interaction'
    ,'Employer_Category' ,'Category1_y',  'Patient_ID_y', 'Category2_y', 'Category3_y',
    'Online_Follower_y' ,'Health Score', 'City_Type2_y', 'City_Type',
    'Number_of_stall_visited', 'Last_Stall_Visited_Number','Patient_ID_y']

    df_ = df.drop(drop_thez, axis =1)
    return df_

#Why_Not_Do_This < because the word 'day' was still rpesent in datetime object?
def scale(df): 
    '''
    Scale columns that are non-ordinal 
    '''
    columnz = ['Var1','Var2', 'Var3', 'Var4', 'Var5', 
       'Camp Start Date - Registration Date',
       'Registration Date - First Interaction',
       'Camp Start Date - First Interaction',
       'Camp End Date - Registration Date', 'Camp Length']

    for i in columnz:
        i_ = df[i].to_frame() 
        
        i_i = scaler.fit_transform(i_)
        df[i] = i_i
    return df  

def one_hot_encoding(df, columns ): #=[ 'online_score','Category 1','Category 2', 'Category 3']
    '''
    Hot encoding of categorical columns 
    '''
    hot_df = df.copy()
    for i in columns:
        if i != 'Category 2':
            dummies = pd.get_dummies(df[i], drop_first=True)
            hot_df = pd.concat([hot_df, dummies[:]], axis=1)
        else:
            dummies = pd.get_dummies(df[i], drop_first=False)
            hot_df = pd.concat([hot_df, dummies[:]], axis=1)

    del hot_df['B']
    # columns2 =['Job Type_x']

    # for i in columns2:
    #     dummies = pd.get_dummies(df[i], drop_first=False)
    #     hot_df = pd.concat([hot_df, dummies[:]], axis=1)

    # del hot_df['Real Estate']
     
    return hot_df 




if __name__ == '__main__':
   # k2_,s2_,xg_=change_cols(k2),change_cols(s2),change_cols(ids)
    combo_df = combine(lst=[k2,s2,xg , ['kNN','sVC','xg']] )
    combo_df_= edit(combo_df)
    #print(combo_df)
    combo_df1 = conditions(combo_df_)
    print(graph(combo_df1))

    combo_df1.to_csv('/home/allen/Galva/capstones/capstone2/data/model_agent.csv')
