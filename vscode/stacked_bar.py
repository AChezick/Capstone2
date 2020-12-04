import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
pd.set_option('display.max_columns', None) 
plt.rcParams['figure.dpi'] = 200

df = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/D7.csv')



cols = ['Online_Follower','2', '3', '4', 'Twitter_Shared','Income', 
       '5', '6', '7', '9999', 'B', 'C',
       'D', 'E', 'F','LinkedIn_Shared',  'G', 'H', 'I',  'Facebook_Shared','Z']

def get_data(x):
    counts_category = {}
    for i in x:
        get = df[i].values
        get_0 = [1 for x in get if x ==0]
        get_1 = [x for x in get if x ==1]

        if i not in counts_category:
            counts_category[i] = (sum(get_0), sum(get_1))

    return counts_category

counts_to_plot = get_data(cols)

 

N = len(cols)
nd = np.arange(N)
no = [v[0] for k,v in counts_to_plot.items()]  
yes = [v[1] for k,v in counts_to_plot.items()]  

ind = np.arange(N)    # the x locations for the groups
width = 0.2       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, no, width)
p2 = plt.bar(ind, yes, width, bottom=no)

plt.ylabel('Counts')
plt.title('Total Attendance by Parameter')
plt.xticks(nd, cols)
plt.yticks(np.arange(0, 45000 , 7500))
plt.legend(('No', 'Yes'))
plt.xticks(rotation=45)
plt.show() 
plt.tight_layout() 

# g = sns.factorplot('Total Attendance by Parameter', data = counts_to_plot.values(), aspect=1.5, kind="count", color=("b", "r")) 
# g.set_xticklabels(rotation=30)
# plt.show() 