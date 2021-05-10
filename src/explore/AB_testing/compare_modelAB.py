
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
pd.set_option('display.max_columns', None) 
plt.figure(dpi=200)

    
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
    
    score = df['y_target'].values
    score_=[]
    for i in score: 
        if i ==1:
            score_.append(1)
        else:
            score_.append(0)
    df['Y_target_SUM'] = df['Y_count_allModels'] + score_ 

    return df
       


if __name__ == '__main__':
   # k2_,s2_,xg_=change_cols(k2),change_cols(s2),change_cols(ids)
    combo_df = combine(lst=[k2,s2,xg , ['kNN','sVC','xg']] )
    combo_df_= edit(combo_df)
