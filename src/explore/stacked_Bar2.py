import pandas as pd 
import numpy as np 
pd.set_option('display.max_columns', None) 
import matplotlib.pyplot as plt 
import seaborn as sns 
plt.rcParams['figure.dpi'] = 200
df = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/ready12_24_train.csv') 
df_feat = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/train_4_model.csv')
from matplotlib import rc
rc('font' )

df_Var1, df_Var5, df_9999, df_23384 = df_feat.copy(),df_feat.copy(),df_feat.copy(),df_feat.copy()
print(df_feat.info(),df_feat.columns)
def make_cols_binary(dataframe): # will also input the column name ***
    '''
    create columns for histogram
    '''
    #print()
    columnz = [ 'B', 'C', 'D', 'E', 'F', 'G'
       ]  
    df_with_colz = dataframe.copy() # create copy of dataframe
    for i in columnz: # for each item in the column

        counter1 =0
        counter2 =1
        
        for ii in ['Yes']: # for each item in list of ranges n
            i_ = str(i)
               # convert value into string for new column title
            title =  i_ 
 
            df_with_colz[title] = df_with_colz[i].apply(lambda x: 1 if x==1 else 0)

    return df_with_colz
                    

def make_cols2(dataframe): # will also input the column name ***
    '''
    create columns for histogram
    '''
# check other function for updating df # 
    columnz = ['Var5']  
     
    for i in columnz: # for each item in the column

        n = list(np.arange(0,320,20)) # for dfirst_start
        counter1 =0
        counter2 =1
        df_with_colz = dataframe.copy() # create copy of dataframe

        for ii in n: # for each item in list of ranges n
            
            r2 = n[1] - n[0] # calculate next range value
            title = (ii + r2) 
            ii_,combo = str(ii),str(title)  # convert current range value into string for new column title
            title_ = ii_ + ' ' + 'to' + ' ' + combo + ' ' + 'Var5'
 
            if counter1 == 0: # check for first value in range - boolean 
                df_with_colz[title_] = dataframe[i].apply(lambda x: 1 if x< counter1 else 0)
            if counter1 > 0:
                ii_ = ii_ + str(r2)
                df_with_colz[title_] = dataframe[i].apply(lambda x: 1 if x< counter1 and x>=counter2 else 0 )

            counter1+=1 # update counter for next item in range
            counter2+=1 # update counter2 for next item in range n

    return df_with_colz


def get_counts(dataframe):
    '''
    get counts for y_target for each col above
    '''
    '''
    '0 to 20 Var5','20 to 40 Var5', '40 to 60 Var5', '60 to 80 Var5', '80 to 100 Var5',
       '100 to 120 Var5', '120 to 140 Var5', '140 to 160 Var5',
       '160 to 180 Var5', '180 to 200 Var5', '200 to 220 Var5',
       '220 to 240 Var5', '240 to 260 Var5', '260 to 280 Var5',
       '280 to 300 Var5', '0 to 1 Var ','2  Var1', '3 Var1','4 to 10 Var1', '11 to 19 Var1',  
            '20 to 39 Var1','40 to 59 Var1', '60 to 79 Var1', '80 to 99 Var1', 
            '100 to 119 Var1','120 to 139 Var1', '140 to 159 Var1'
        '9999.0', '23384', 'Third','1036', '1216', '1217', '1352',
       '1704', '1729', '2517', '2662', '23384',
    '''
    lst =  [  'B', 'C', 'D', 'E', 'F', 'G',
       ]  

    lst2 = ['9999.0', '23384', 'Third','1036', '1216', '1217', '1352',
       '1704', '1729', '2517', '2662', '23384', 'B', 'C', 'D', 'E', 'F', 'G',
       '2100','Second','Third']
    nums = []
    df_ = dataframe 
    if df_.shape == df_Var5.shape:
        lst_ = lst
    else:
        lst_ = lst2
        
    for i in lst_:
        get_0 = df_.loc[(df_['y_target'] == 0) & (df_[i] == 1 ) ] 
        get_1 = df_.loc[ (df_['y_target'] == 1) & (df_[i] == 1 ) ] 
        ap = (i,( len(get_0), len(get_1)) )
        nums.append(ap)
     
    return nums

def create_graph(dataframe): # as of 1/3/21 mostly fixed output - but need to decide on how / which to display
    '''                                      # seems like doing the Vars separate would be better 
    Stacked bar chart 
    '''
    countz = get_counts(dataframe)
    bars1 = [x[1][1] for x in countz ] 
    bars2 = [x[1][0] for x in countz ] 

    bars = np.add(bars1, bars2).tolist()
    r = [str(i) for i in bars1]
    
    names = [x[0] for x in countz] # The names for each group 
    
    barWidth = 1
    plt.bar(r, bars1, color= '#dd6101', edgecolor='white', width=barWidth)
    # Create green bars (middle), on top of the firs ones
    plt.bar(r, bars2, bottom=bars1, color='#557f2d', edgecolor='white', width=barWidth)
    
    # Custom axis
    plt.title(label = 'Attendance Counts among Encoded Features')
    plt.xticks(r, names )
    plt.xlabel("group")
    plt.xticks(rotation=75) 
    plt.xlabel("Feature")
    plt.ylabel("Number of Patients")
    plt.legend( ('Did Attend', 'Did NOT Attend')) 
    # Show graphic
    plt.show()
    return plt.show()   
    


if __name__ == '__main__':
    colz_9999 = make_cols_binary(df_9999) #, make_cols_binary(df_23384)


    #print(colz_9999.columns )
    
    print(create_graph(colz_9999))
    #colz_3_graph , colz_99_graph= create_graph(colz3 ) ,create_graph(colz_9999 ) 
    #colz_Var1, colz_Var5 = make_cols_binary(df_Var1) , make_cols_binary(df_Var2)
    #print(colz_99_graph , colz_3_graph)
    # , colz_23384 













# def make_inter_reg(dataframe):
#     '''
#     specific for unique range for the feature interaction_reg_delta
#     '''
#     df2_ = dataframe.copy() 
     
#     df2_['0 to 1 Var1'] =  df2_['Var1'].apply( lambda x: 1 if x <=1 else 0)
#     df2_['2  Var1'] =  df2_['Var1'].apply( lambda x: 1 if x == 2 else 0)
#     df2_['3 Var1'] =  df2_['Var1'].apply( lambda x: 1 if x == 3 else 0)
#     df2_['4 to 10 Var1'] =  df2_['Var1'].apply( lambda x: 1 if x <=10 and x>= 4 else 0)
#     df2_['11 to 19 Var1'] =  df2_['Var1'].apply( lambda x: 1 if x <=19 and x>= 11 else 0)
#     df2_['20 to 39 Var1'] =  df2_['Var1'].apply( lambda x: 1 if x <39 and x>= 20 else 0)
#     df2_['40 to 59 Var1'] =  df2_['Var1'].apply( lambda x: 1 if x <59 and x>= 40 else 0)
#     df2_['60 to 79 Var1'] =  df2_['Var1'].apply( lambda x: 1 if x <79 and x>=60 else 0)
#     df2_['80 to 99 Var1'] =  df2_['Var1'].apply( lambda x: 1 if x <99 and x>=80 else 0)
#     df2_['100 to 119 Var1'] =  df2_['Var1'].apply( lambda x: 1 if x <119 and x>=100 else 0)
#     df2_['120 to 139 Var1'] =  df2_['Var1'].apply( lambda x: 1 if x <139 and x>= 120 else 0)
#     df2_['140 to 159 Var1'] =  df2_['Var1'].apply( lambda x: 1 if x <159 and x>= 140 else 0)
     
#     df2_['160 to 179 Var1'] =  df2_['Var1'].apply( lambda x: 1 if x <179 and x>= 160 else 0)
#     df2_['180 to 199 Var1'] =  df2_['Var1'].apply( lambda x: 1 if x <199 and x>=180 else 0)
#     df2_['200 to 219 Var1'] =  df2_['Var1'].apply( lambda x: 1 if x <219 and x>=200 else 0)
#     df2_['220 to 239 Var1'] =  df2_['Var1'].apply( lambda x: 1 if x <239 and x>=220 else 0)
#     df2_['240 to 259 Var1'] =  df2_['Var1'].apply( lambda x: 1 if x <259 and x>= 240 else 0)
#     df2_['260 to 279 Var1'] =  df2_['Var1'].apply( lambda x: 1 if x <279 and x>= 260 else 0)
#     df2_['280 to 299 Var1'] =  df2_['Var1'].apply( lambda x: 1 if x <299 and x>= 280 else 0)
