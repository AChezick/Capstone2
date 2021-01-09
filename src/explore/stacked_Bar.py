import pandas as pd 
import numpy as np 
pd.set_option('display.max_columns', None) 
import matplotlib.pyplot as plt 
import seaborn as sns 
plt.rcParams['figure.dpi'] = 200
df = pd.read_csv('/home/allen/Galva/capstones/capstone2/src/explore/ready12_24_train.csv') 
from matplotlib import rc
rc('font', weight='bold')

df_delta_first_reg, df_delta_reg_end, df_Camp_Length, df_interaction_regreister_delta = df.copy() , df.copy() ,df.copy(), df.copy()

def make_cols1(dataframe): # will also input the column name ***
    '''
    create columns for histogram
    '''
# check other function for updating df # 

    columnz = ['delta_reg_end'] # *** will be 

    for i in columnz: # for each item in the column

        n = list(np.arange(0,500,20)) # for dfirst_start
        counter1 =0
        counter2 =1
        df_with_colz = dataframe.copy() # create copy of dataframe

        for ii in n: # for each item in list of ranges n
            
            r2 = n[1] - n[0] # calculate next range value
            title = (ii + r2) 
            ii_,combo = str(ii),str(title)  # convert current range value into string for new column title
            title = ii_ + ' ' + 'to' + ' ' + combo
 
            if counter1 == 0: # check for first value in range - boolean 
                df_with_colz[title] = dataframe[i].apply(lambda x: 1 if x< counter1 else 0)
            if counter1 > 0:
                ii_ = ii_ + str(r2)
                df_with_colz[title] = dataframe[i].apply(lambda x: 1 if x< counter1 and x>=counter2 else 0 )

            counter1+=1 # update counter for next item in range
            counter2+=1 #update counter2 for next item in range n

    return df_with_colz

def make_cols2(dataframe): # will also input the column name ***
    '''
    create columns for histogram
    '''
# check other function for updating df # 
    columnz = ['delta_first_reg'] # *** will be 
     
    for i in columnz: # for each item in the column

        n = list(np.arange(-500,4500,250)) # for dfirst_start
        counter1 =0
        counter2 =1
        df_with_colz = dataframe.copy() # create copy of dataframe

        for ii in n: # for each item in list of ranges n
            
            r2 = n[1] - n[0] # calculate next range value
            title = (ii + r2) 
            ii_,combo = str(ii),str(title)  # convert current range value into string for new column title
            title_ = ii_ + ' ' + 'to' + ' ' + combo
 
            if counter1 == 0: # check for first value in range - boolean 
                df_with_colz[title_] = dataframe[i].apply(lambda x: 1 if x< counter1 else 0)
            if counter1 > 0:
                ii_ = ii_ + str(r2)
                df_with_colz[title_] = dataframe[i].apply(lambda x: 1 if x< counter1 and x>=counter2 else 0 )

            counter1+=1 # update counter for next item in range
            counter2+=1 #update counter2 for next item in range n

    return df_with_colz

def make_cols3(dataframe): # will also input the column name ***
    '''
    create columns for histogram
    '''
# check other function for updating df # 
    columnz = ['delta_first_reg'] # *** will be 
     
    for i in columnz: # for each item in the column

        n = list(np.arange(0,800,25)) # for dfirst_start
        counter1 =0
        counter2 =1
        df_with_colz = dataframe.copy() # create copy of dataframe

        for ii in n: # for each item in list of ranges n
            
            r2 = n[1] - n[0] # calculate next range value
            title = (ii + r2) 
            ii_,combo = str(ii),str(title)  # convert current range value into string for new column title
            title = ii_ + ' ' + 'to' + ' ' + combo
 
            if counter1 == 0: # check for first value in range - boolean 
                df_with_colz[title] = dataframe[i].apply(lambda x: 1 if x< counter1 else 0)
            if counter1 > 0:
                ii_ = ii_ + str(r2)
                df_with_colz[title] = dataframe[i].apply(lambda x: 1 if x< counter1 and x>=counter2 else 0 )

            counter1+=1 # update counter for next item in range
            counter2+=1 #update counter2 for next item in range n

    return df_with_colz

def make_inter_reg(dataframe):
    '''
    specific for unique range for the feature interaction_reg_delta
    '''
    df_ = dataframe
    df_['-4000 to -3000'] =  df_['interaction_regreister_delta'].apply( lambda x: 1 if x<-3000 else 0)
    df_['-2999 to 0'] =  df_['interaction_regreister_delta'].apply( lambda x: 1 if x < 0 and x>=-300 else 0)

    df_['0 to 14'] =  df_['interaction_regreister_delta'].apply( lambda x: 1 if x <14 and x>= 0 else 0)
    df_['15 to 30'] =  df_['interaction_regreister_delta'].apply( lambda x: 1 if x <30 and x>= 14 else 0)
    df_['31 to 49'] =  df_['interaction_regreister_delta'].apply( lambda x: 1 if x <50 and x>= 30 else 0)

    df_['50 to 99'] =  df_['interaction_regreister_delta'].apply( lambda x: 1 if x <100 and x>=49 else 0)
    df_['100 to 199'] =  df_['interaction_regreister_delta'].apply( lambda x: 1 if x <200 and x>=100 else 0)
    df_['200 to 399'] =  df_['interaction_regreister_delta'].apply( lambda x: 1 if x <400 and x>=200 else 0)
    df_['400 to 599'] =  df_['interaction_regreister_delta'].apply( lambda x: 1 if x <600 and x>=400 else 0)
    df_['600 to 799'] =  df_['interaction_regreister_delta'].apply( lambda x: 1 if x <800 and x>=600 else 0)
    df_['800 to 999'] =  df_['interaction_regreister_delta'].apply( lambda x: 1 if x <1000 and x>=800 else 0)
    df_['1000 +'] =  df_['interaction_regreister_delta'].apply( lambda x: 1 if x>=1000 else 0)

    return df_ 

def get_counts(dataframe):
    '''
    get counts for y_target for each column in the list
    '''
    lst = list(dataframe.columns)
    print(lst)
    nums = [] 
    df_ = dataframe
    for i in lst:
        # if i != 'City_Type':
        get_0 = df_.loc[(df_['y_target'] == 0) & (df_[i] == 1 ) ] 
        get_1 = df_.loc[ (df_['y_target'] == 1) & (df_[i] == 1 ) ] 
        ap = (i,( len(get_0), len(get_1)) )
        nums.append(ap)
        # else: #City Type is non-binary 
        #     get_0 = df_.loc[(df_['y_target'] == 0) & (df_[i] != 1 ) ] 
        #     get_1 = df_.loc[ (df_['y_target'] == 1) & (df_[i] != 1 ) ] 
        #     ap = (i,( len(get_0), len(get_1)) )
        #     nums.append(ap)


    return nums

def create_graph(dataframe): # as of 1/1/21 need to edit which columns are being shown in output graph 
    '''
    Stacked bar chart 
    '''
    to_dropp = ['Patient_ID_x', 'Health_Camp_ID', 'Registration_Date', 
    'Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Category1_x', 'Category2', 
    'Category3', 'Camp_Start_Date2', 'Camp_End_Date2', 'patient_event', 
    'Unnamed: 0_x', 'Unnamed: 0.1_x', 'Online_Follower_x', 'First_Interaction', 'Event1_or_2_x', 'online_score', 
    'Category1_y', 'Unnamed: 0_y', 'Unnamed: 0.1_y', 'Patient_ID_y', 'Online_Follower_y', 
    'Event1_or_2_y', 'Health Score', 'Number_of_stall_visited', 'Last_Stall_Visited_Number', 
    'Camp_length', 'delta_first_reg', 'interaction_regreister_delta', 'delta_first_start', 'delta_reg_end', 
    'Camp_Length', ] 
    

    dataframe_  = dataframe.drop(to_dropp, axis=1)
    countz = get_counts(dataframe_)
     
    bars1 = [x[1][1] for x in countz] 
    bars2 = [x[1][1] for x in countz] 

    bars = np.add(bars1, bars2).tolist()
    r = [str(i) for i in bars1  ]
    print(r, len(r))
    names = [x[0] for x in countz ] # The names for each group 
    
    barWidth = 1
    plt.bar(r, bars1, color= '#dd6101', edgecolor='white', width=barWidth)
    # Create green bars (middle), on top of the firs ones
    plt.bar(r, bars2, bottom=bars1, color='#557f2d', edgecolor='white', width=barWidth)
    
    # Custom X axis
    plt.title(label = 'City Location Splits')
    #plt.xticks(r , names, fontweight='bold')
    plt.xlabel("Feature")
    plt.ylabel("Number of Patients")
    plt.xticks(rotation=65) 
    plt.legend( ('Did Attend', 'Did NOT Attend')) 
    # Show graphic
    plt.show()
    return plt.show()   
    


if __name__ == '__main__':
    colz_first_reg = make_cols1(dataframe = df_delta_reg_end ) 
    '''call this f(x) for each graph to be made''' 
    #colz_delta_first_reg = make_cols2(dataframe = df_delta_first_reg)

    # colz_delt_reg_end = make_cols2(dataframe = df_delta_reg_end)
    # colz_int_reg_delta = make_inter_reg(df_interaction_regreister_delta) 
    # first_try =  create_graph(colz_int_reg_delta)
     
    # something = make_cols3(df_Camp_Length)

    print(create_graph(colz_first_reg))
'''
'-500 to -250', '-250 to 0', '0 to 250', '250 to 500', '500 to 750', '750 to 1000', 
    '1000 to 1250', '1250 to 1500', '1500 to 1750', '1750 to 2000', '2000 to 2250', '2250 to 2500', 
    '2500 to 2750', '2750 to 3000', '3000 to 3250', '3250 to 3500', '3500 to 3750', '3750 to 4000', 
    '4000 to 4250', '4250 to 4500'
'''