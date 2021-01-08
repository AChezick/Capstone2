import sqlite3 
from pathlib import Path
Path('my_data.db').touch()
conn = sqlite3.connect('my_data.db')
c = conn.cursor() 
import pandas as pd 
import numpy as np 

# c.execute('''CREATE TABLE third1 (Patient_ID int, Health_Camp_ID int, Number_of_stall_visited int,Last_Stall_Visited_Number int)''')  

# third1=pd.read_csv('/home/allen/Galva/capstones/capstone2/data/Train/Third_Health_Camp_Attended.csv')
# third1.to_sql('third1', conn, if_exists='append',index=False)

print(third1 = c.execute('''
SELECT *
FROM second_
LIMIT 10;
''').fetchall() ) 

check_twoes 
