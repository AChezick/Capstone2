import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
pd.set_option('display.max_columns', None) 
plt.rcParams['figure.dpi'] = 200

df = pd.read_csv('/home/allen/Galva/capstones/capstone2/data/D7.csv')

print(df.columns)

cols = ['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Patient_ID',
       'Online_Follower', 'LinkedIn_Shared', 'Twitter_Shared',
       'Facebook_Shared', 'Income', 'Education_Score', 'Age',
       'First_Interaction', 'City_Type', 'Employer_Category', 'Job_Type',
       'Event1_or_2', 'Education_Score2', 'Education_Scorez', '2', '3', '4',
       '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '9999', 'B', 'C',
       'D', 'E', 'F', 'G', 'H', 'I', 'Z'] 

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
print(counts_to_plot)
 

# N = len(cols)
# nd = np.arange(N)
# no = [v[0] for k,v in counts_to_plot.items()]  
# yes = [v[1] for k,v in counts_to_plot.items()]  

# ind = np.arange(N)    # the x locations for the groups
# width = 0.2       # the width of the bars: can also be len(x) sequence

# p1 = plt.bar(ind, no, width)
# p2 = plt.bar(ind, yes, width, bottom=no)

# plt.ylabel('Counts (Thousands)')
# plt.title('Total Attendance by Parameter')
# #plt.xticks(nd, cols)
# plt.yticks(np.arange(0, 45000 , 7500))
# plt.legend(('No', 'Yes'))
# #plt.xticks(rotation=45)
# plt.show() 
# plt.tight_layout() 

for i,ii in counts_to_plot:

    fig, axes  = plt.subplots(2,4)

    axes[0].bar(i,(ii[0], ii[1]) , aligh='center', width = .5, alpha =.5)
    set_title("Total Attendance by Parameter") 