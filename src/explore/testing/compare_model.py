
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
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

def conditions(df):
    y_target = df['y_target'].values
    score = df['Y_count'].values
    score_=[]
    for i in score:
        if i ==0:
            score_.append(1)
        else:
            score_.append(0)
    df['Y_target2'] = df['Y_count'] + score_

    return df
       
def edit(df): 
    
    new_df = df.copy()
    new_df['Y_count'] = df['prediction'] + df['prediction_k2'] + df['prediction_s2']+ df['prediction_xg']
    #new_df['Y_count'].apply(lambda x: x+1 if df['y_target'] == 0 else x)
    # new_df.loc[(new_df['y_target'] ==0) & (new_df['y_Count']==3) ]
#     new_df['all_models'] = np.where(
#         df['prediction_k2']==df['prediction_s2'] ==df['prediction_xg'] == df['y_target']
#    ,4  )   
    
    # could do a sum across rows for y_predict value : 0,1,2,3,4
    # if its a 2 then check for ac,ab,bc
    # if its a 1 then check for a,b,c 
    # assign letter or number based on 1-6
    return new_df 

    
def graph(new_df):
    to_print = pd.DataFrame({

        'knn_predict' :new_df['prediction_k2'].values ,
        'xg_predict'  : new_df['prediction_xg'].values ,
        'src_predict ' : new_df['prediction_k2'].values,
        'knn_proba' : new_df['proba_k2'].values,
        'xg_proba'  : new_df['proba_xg'].values,
        'src_proba' : new_df['proba_k2'].values
    })
    #lines = to_print.plot.line()
    #plt.hist(x=new_df['Y_count'])
    g = sns.relplot(data = new_df, x='Y_target2', y ='Y_count', col='y_target', palette = 'deep', kind='scatter')
      
    plt.show() 
    return None 



if __name__ == '__main__':
   # k2_,s2_,xg_=change_cols(k2),change_cols(s2),change_cols(ids)
    combo_df = combine(lst=[k2,s2,xg , ['k2','s2','xg']] )
    combo_df_=edit(combo_df)
    print(combo_df)
    combo_df1 = conditions(combo_df_)
    print(graph(combo_df1))

    combo_df1.to_csv('/home/allen/Galva/capstones/capstone2/data/model_agent.csv')
