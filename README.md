# Capstone2

The goal of my project was to help MedCamp -a healthfair provider company – reduce wasteful spending AND maintain quality experiences of attendies by accuretly predcting who will and will not attend a healthfare events.

--- 

## Data 
This project, Healthcare Analytics, came from [Kaggle](https://www.kaggle.com/vin1234/janatahack-healthcare-analytics?select=Train) 

##### Kaggle Description 

train.zip contains 6 different csv files apart from the data dictionary as described below:

**Health_Camp_Detail.csv** – File containing HealthCampId, CampStartDate, CampEndDate and Category details of each camp.

**Train.csv** – File containing registration details for all the test camps. This includes PatientID, HealthCampID, RegistrationDate and a few anonymized variables as on registration date.

---

|   | Patient_ID | Health_Camp_ID | Registration_Date | Var1 | Var2 | Var3 | Var4 | Var5 |
|---|------------|----------------|-------------------|------|------|------|------|------|
| 0 | 489652     | 6578           | 10-Sep-05         | 4    | 0    | 0    | 0    | 2    |
| 1 | 507246     | 6578           | 18-Aug-05         | 45   | 5    | 0    | 0    | 7    |
| 2 | 523729     | 6534           | 29-Apr-06         | 0    | 0    | 0    | 0    | 0    |
| 3 | 524931     | 6535           | 07-Feb-04         | 0    | 0    | 0    | 0    | 0    |
| 4 | 521364     | 6529           | 28-Feb-06         | 15   | 1    | 0    | 0    | 7    | 

--- 

**Patient_Profile.csv** – This file contains Patient profile details like PatientID, OnlineFollower, Social media details, Income, Education, Age, FirstInteractionDate, CityType and EmployerCategory

--- 
|   | Patient_ID | Online_Follower | LinkedIn_Shared | Twitter_Shared | Facebook_Shared | Income | Education_Score | Age | First_Interaction | City_Type | Employer_Category |
|---|------------|-----------------|-----------------|----------------|-----------------|--------|-----------------|-----|-------------------|-----------|-------------------|
| 0 | 516956     | 0               | 0               | 0              | 0               | 1      | 90              | 39  | 18-Jun-03         |           | Software Industry |
| 1 | 507733     | 0               | 0               | 0              | 0               | 1      | None            | 40  | 20-Jul-03         | H         | Software Industry |
| 2 | 508307     | 0               | 0               | 0              | 0               | 3      | 87              | 46  | 02-Nov-02         | D         | BFSI              |
| 3 | 512612     | 0               | 0               | 0              | 0               | 1      | 75              | 47  | 02-Nov-02         | D         | Education         |
| 4 | 521075     | 0               | 0               | 0              | 0               | 3      | None            | 80  | 24-Nov-02         | H         | Others            

 ---      

**First_Health_Camp_Attended.csv** – This file contains details about people who attended health camp of first format. This includes Donation (amount) & Health_Score of the person.

--- 

|   | Patient_ID | Health_Camp_ID | Donation | Health_Score | Unnamed: 4 |
|---|------------|----------------|----------|--------------|------------|
| 0 | 506181     | 6560           | 40       | 0.43902439   |            |
| 1 | 494977     | 6560           | 20       | 0.097560976  |            |
| 2 | 518680     | 6560           | 10       | 0.048780488  |            |
| 3 | 509916     | 6560           | 30       | 0.634146341  |            |
| 4 | 488006     | 6560           | 20       | 0.024390244  |            |

---

**Second_Health_Camp_Attended.csv** - This file contains details about people who attended health camp of second format. This includes Health_Score of the person.

**Third_Health_Camp_Attended.csv** - This file contains details about people who attended health camp of third format. This includes Numberofstallvisited & LastStallVisitedNumber.

**Test.csv** – File containing registration details for all the camps done after 1st April 2006. This includes PatientID, HealthCampID, RegistrationDate and a few anonymized variables as on registration date. Participant should make predictions for these patient camp combinations


### EDA

There were imbalanced classes among potential health fair attendies. However, this ratio was propotionally split and the train test ratios were smaller.
There were 18 NO, and 16000 in the yes. Thus 8,000 patients were in both the train and test group provided by the company. 

|              | Total | Train | Test  | Overlap (Multi-Attends)  |
|--------------|-------|-------|-------|--------------------------|
| Yes Attended | 26565 | 45275 | 16743 | 28532                    |
| No           | 11068 | 30003 | 18506 | 11497                    |


show attendies 0/ 1

The additional features provided by MedCamp were deffenitally imbalanced and colinear 

show hist plot



---


### Modeling 

[[16088  2418]]
 [[ 9349  7394]]

 Specifically reducing false positives and negaitves. 
 
---


### Discussion 


---


### Next Steps
