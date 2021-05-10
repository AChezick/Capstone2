
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



if __name__ == '__main__':
   # k2_,s2_,xg_=change_cols(k2),change_cols(s2),change_cols(ids)
    combo_df = combine(lst=[k2,s2,xg , ['kNN','sVC','xg']] )
    combo_df_= edit(combo_df)
    #print(combo_df)
    combo_df1 = conditions(combo_df_)
    print(graph(combo_df1))

    combo_df1.to_csv('/home/allen/Galva/capstones/capstone2/data/model_agent.csv')
