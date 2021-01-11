# Are you going? I am not sure ... Lets think about it



MedCamp is a non-profit which provides health camps for people in various cities acorss America.  MedCamp was having challenges maintaining their operations due to excessive operational expense. 

#### Goal 
Help MedCamp reduce wasteful spending AND maintain quality experiences of attendees by accurately predicting who will and will not attend a health-fare events. 

--- 

## Data 
This project, Healthcare Analytics, came from [Kaggle](https://www.kaggle.com/vin1234/janatahack-healthcare-analytics?select=Train) 

##### Kaggle Description 

train.zip contains 6 different csv files apart from the data dictionary as described below:

**Health_Camp_Detail.csv** – File containing HealthCampId, CampStartDate, CampEndDate and Category details of each camp.

---

| Health_Camp_ID | Camp_Start_Date | Camp_End_Date | Category1 | Category2 | Category3 |
|----------------|-----------------|---------------|-----------|-----------|-----------|
| 6560           | 16-Aug-03       | 20-Aug-03     | First     | B         | 2         |
| 6530           | 16-Aug-03       | 28-Oct-03     | First     | C         | 2         |
| 6544           | 03-Nov-03       | 15-Nov-03     | First     | F         | 1         |
| 6585           | 22-Nov-03       | 05-Dec-03     | First     | E         | 2         |
| 6561           | 30-Nov-03       | 18-Dec-03     | First     | E         | 1         |

---

**Train.csv** , **Test.csv** – Both files have similar layouts, containing registration details for all the test camps. This includes PatientID, HealthCampID, RegistrationDate and a few anonymized variables as on registration date. Test.csv – File containing registration details for all the camps done after 1st April 2006. This includes PatientID, HealthCampID, RegistrationDate and a few anonymized variables as on registration date. 

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

**First_Health_Camp_Attended.csv** & **Second_Health_Camp_Attended.csv** – These files contain details about people who attended health camp of first format. This includes Donation (amount) & Health_Score of the person.

--- 

|   | Patient_ID | Health_Camp_ID | Donation | Health_Score | 
|---|------------|----------------|----------|--------------|
| 0 | 506181     | 6560           | 40       | 0.43902439   |            
| 1 | 494977     | 6560           | 20       | 0.097560976  |            
| 2 | 518680     | 6560           | 10       | 0.048780488  |            
| 3 | 509916     | 6560           | 30       | 0.634146341  |            
| 4 | 488006     | 6560           | 20       | 0.024390244  |            

---

 - This file contains details about people who attended health camp of second format. This includes Health_Score of the person.

**Third_Health_Camp_Attended.csv** - This file contains details about people who attended health camp of third format. This includes Numberofstallvisited & LastStallVisitedNumber.

--- 

| Patient_ID | Health_Camp_ID | Number_of_stall_visited | Last_Stall_Visited_Number |
|------------|----------------|-------------------------|---------------------------|
| 517875     | 6527           | 3                       | 1                         |
| 504692     | 6578           | 1                       | 1                         |
| 504692     | 6527           | 3                       | 1                         |
| 493167     | 6527           | 4                       | 4                         |
| 510954     | 6528           | 2                       | 2                         |

---



## EDA

There were imbalanced classes among potential health camp attendees. Additionally, it is important to note that some patients attended more than one MedCamp health event.  

--- 

| 37633   | Unique Patient IDs                                 |
|---------|----------------------------------------------------|
| 65      | Unique Health Camps                                |
| 20,534  | Count of Patients Attending a Health Camp          |
| 15,011  | Unique Patients Attending at least one Health Camp |
| 102,000 | Patient-Event Registrations                        |
| ~ 20%   | Historic Attendance Rate                           |
| 3       | Classes or Types of Health Camps                   |

---

### Creating Target Variable Y

According to the description on Kaggle ,MedCamp wanted to know the probability that a patient would successfully attend a health-fair event. For the first two camp types success was defined as getting a health score. For the third event-type success was going to at least one booth. 

#### Primary Key

Given that each patient could attend more than one event, it was necessary to create a primary key for each patient & Health Camp combination by concatenating of the Patient and Camp ID.  

| Health Camp ID 6578 | Patient ID 489652  | Primary Key 4896526578  |
|---------------------|--------------------|-------------------------|

Creating this primary key was helpful in combining information and creating additional time features; meaningful data was spread among several csv files. 
![]( https://github.com/AChezick/Capstone2/blob/main/images/images2/primary_key.png ) 


### Feature Engineering 

Training the model with only the five anonymized features results in very poor performance.

![]( https://github.com/AChezick/Capstone2/blob/main/images/non_feature_all_models.png )

For the two anonymized features (Var1 , Var5) that had the highest feature weights, most of the counts were zero value. 

![]( https://github.com/AChezick/Capstone2/blob/main/images/images2/var1_5_Zero.png ) 

For comparison here is the rest of the distribution for Var1 & Var5.  Without knowing what either of these features map to, I decided not to drop them for modeling purposes.  

![]( https://github.com/AChezick/Capstone2/blob/main/images/images2/non_zerovar1_5.png ) 

Thus, feature engineering was instrumental in improving the model. 

#### One Hot Encoding

I used one hot enocding on several of the categorical features. Several of the models reported 9999.0 - as a  result from a one-hot encoding. 

--- 

| Var1 | Var2 | Var3 | Var4 | Var5 | y_target | Camp Start Date - Registration Date | Registration Date - First Interaction | Camp Start Date - First Interaction | Camp End Date - Registration Date | Camp Length | 1036 | 1216 | 1217 | 1352 | 1704 | 1729 | 2517 | 2662 | 23384 | Second | Third | B | C | D | E | F | G | 2100 | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 | 7.0 | 8.0 | 9.0 | 10.0 | 11.0 | 12.0 | 13.0 | 14.0 | 9999.0 | 1 | 2 | 3 | 4 |
|------|------|------|------|------|----------|-------------------------------------|---------------------------------------|-------------------------------------|-----------------------------------|-------------|------|------|------|------|------|------|------|------|-------|--------|-------|---|---|---|---|---|---|------|-----|-----|-----|-----|-----|-----|-----|-----|------|------|------|------|------|--------|---|---|---|---|
| 4.0  | 0.0  | 0.0  | 0.0  | 2.0  | 1.0      | -25.0                               | 278                                   | 253                                 | 34                                | 59          | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 1     | 0      | 1     | 0 | 0 | 0 | 0 | 0 | 1 | 1    | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0    | 0    | 0    | 0    | 0    | 1      | 0 | 0 | 0 | 0 |
| 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0      | -24.0                               | 99                                    | 75                                  | 161                               | 185         | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 1     | 0      | 0     | 0 | 0 | 0 | 0 | 1 | 0 | 1    | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0    | 0    | 0    | 0    | 0    | 1      | 0 | 0 | 0 | 0 |
| 4.0  | 0.0  | 0.0  | 0.0  | 2.0  | 0.0      | -60.0                               | 355                                   | 295                                 | 711                               | 771         | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 1     | 0      | 0     | 0 | 0 | 0 | 0 | 1 | 0 | 1    | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0    | 0    | 0    | 0    | 0    | 1      | 0 | 0 | 0 | 0 |

--- 

The additional features provided by MedCamp were imbalanced, co-linear, or missing. For example Age, Income, Education Score had less than 2,000 values and most patients who had one of these features had the other 3. Therefore, imputing the average value onto the other 35,000 patients would not be helpful. 

![]( https://github.com/AChezick/Capstone2/blob/main/images/images2/attendance_catgegorical.png ) 


### Features from Dates

I used the primary key to track the unique patient events and consolidate important information into csv that could be used for training and testing. The following features were created:

|   | Feature Name (Days)                   |
|---|---------------------------------------|
|   | Registration Date - First Interaction |
|   | Camp Start Date - Registration Date   |
|   | Camp Start Date - First Interaction   |
|   | Camp End Date - Registration Date     |
|   | Camp Length                           | 


---


### Modeling 

Given the goal is to ensure all patients have a good experience , there has to be extra supplies. However, having accurate predictions means we can be confident in having just enough extra supplies thus covering both goals. 

--- 

#### Results after creating features, one-hot encoding, scaling 

![]( https://github.com/AChezick/Capstone2/blob/main/images/roc_all%20models.png ) 

As shown above all models achelved a similar ROC score. 

![]( https://github.com/AChezick/Capstone2/blob/main/images/images2/pde_campLength.png ) 

The Date features ended up improving scores for all models. Additionally, for all but some iterations of Random Forests, the date/times features would show among top feature importances. 

![]( https://github.com/AChezick/Capstone2/blob/main/images/feature_core_rough.png ) 

---
 

### 

#### Future Work
Since this data is not descriptive black-box models are OK to use. Optimization of a neural network may produce good results. I used tensorflow and keras and was able to achieve similar results to other models with minimal training. 

![]( https://github.com/AChezick/Capstone2/blob/main/images/images2/BasicNN.png )

However, I am confident that these scores can improve by using a grid search and other optimization techniques. 

![]( https://github.com/AChezick/Capstone2/blob/main/images/images2/BasicNN_loss.png ) 
