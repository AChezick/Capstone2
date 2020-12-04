# Are you going? I am not sure ... 

The goal of my project was to help MedCamp -a health-fare provider company – reduce wasteful spending AND maintain quality experiences of attendees by accurately predicting who will and will not attend a health-fare events. 

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

There were imbalanced classes among potential health fair attendees. However, this ratio was more balanced in split and the train test ratios were smaller.
There were 18 NO, and 16000 in the yes. Thus 8,000 patients were in both the train and test group provided by the company.  

| 37633 Patients | Total | Train | Test  | Overlap (Multi-Attends)  |
|----------------|-------|-------|-------|--------------------------|
| Yes Attended   | 26565 | 45275 | 16743 | 28532                    |
| No             | 11068 | 30003 | 18506 | 11497                    | 

![]( https://github.com/AChezick/Capstone2/blob/main/images/attendance_counts.png ) 

The additional features provided by MedCamp were imbalanced and co-linear  
![]( https://github.com/AChezick/Capstone2/blob/main/images/all_stacked_bar1.png ) 

Additionally most columns are sparse. 

![]( https://github.com/AChezick/Capstone2/blob/main/images/hist_online_features.png ) 


---


### Modeling 

Given my goal is to ensure all patients have a good experience , there has to be extra supplies. However, having accurate predictions means we can be confident in having just enough extra supplies thus covering both goals. I compared 

--- 

#### Results from the Logistic Regression after creating features, one-hot encoding, scaling 

| Random Forest 200 Trees | Accuracy | Precision | Recall | f1-Score |
|--------------------------------|----------|-----------|--------|----------|
| NO Attendance                  | .62      | .69       | .49    | .57      |
| Yes Attendance                 | .62      | .57       | .76    | .65      | 


| Logistic Regression alpha = .56 | Accuracy | Precision | Recall | f1-Score |
|---------------------------------|----------|-----------|--------|----------|
| NO Attendance                   | .65      | .61       | .92    | .74      |
| Yes Attendance                  | .65      | .80       | .36    | .49      |


#### Results from PCA 
![]( https://github.com/AChezick/Capstone2/blob/main/images/PCA_7.png ) 

The most important components were columns 1,2,4,5,24. 

The raw data has no information on what columns 1,2,4,5 are or what the numbers mean. Column 24 points to a city of one of the heathfairs.
However, knowing these components might be important for MedCamp.

---
 

### Next Steps
There are more features which can be extracted from the raw data which might be helpful. Specifically aspects of when someone signed up for an event and if they attended. 

#### Models to try
Since this data is not descriptive black-box models are OK to use. Implementing and optimization a neural network is likely to produce good results. Learning how to use XDGBoost is also likely to improve prediction score. 
