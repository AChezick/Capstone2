# Are you going? I am not sure ... Lets think about it



MedCamp is a non-profit which provides health camps for people in various cities acorss America.  MedCamp was having challenges maintaining their operations due to excessive operational expense. 

#### Goal
Help MedCamp reduce wasteful spending AND maintain quality experiences of attendees by accurately predicting who will and will not attend a health-fare events. 

--- 

## Data 
This project, Healthcare Analytics, came from [Kaggle](https://www.kaggle.com/vin1234/janatahack-healthcare-analytics?select=Train) 

#### Anonymized Features: All data was anonymized

Protecting patient data is critical. However, it does make following this READ.md more difficult. I will reorient the reader throughout! 

---

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

**Train.csv** & **Test.csv** – Both files have similar layouts, containing registration details for all the test camps. This includes PatientID, HealthCampID, RegistrationDate and a few anonymized variables as on registration date. Test.csv – File containing registration details for all the camps done after 1st April 2006. This includes PatientID, HealthCampID, RegistrationDate. 

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

There were imbalanced classes among potential health camp attendees; specifically for each geographic location and among camps occurring within that location showed... all had different attendance rates. Thus, simply estimating attendance based on a global or local history would lead to poor results. Additionally, it is important to note that some patients attended more than one MedCamp health event.  

--- 

| 37633   | Unique Patient IDs                                 |
|---------|----------------------------------------------------|
| 65      | Unique Health Camps                                |
| 20,534  | Count of Patients Attending a Health Camp          |
| 15,011  | Unique Patients Attending at least one Health Camp |
| 102,000 | Patient-Event Registrations                        |
| ~ 20%   | Global Attendance Rate                             |
| 3       | Classes or Types of Health Camps                   |

---

### Creating Target Variable Y

According to the description on Kaggle, MedCamp wanted to know the probability that a patient would successfully attend a health-fair event. For the first two camp types success was defined as getting a health score. For the third event-type success was going to at least one booth. The data from MedCamp was from several years and preliminary EDA showed that each patient could attend more than one Camp. Thus, to correctly create a target feature I needed to know the Camp ID,Patient ID, and if they successfully went to that event.

#### Primary Key

Given that each patient could attend more than one event, it was necessary to create a primary key for each patient & Health Camp combination by concatenating of the Patient and Camp ID.  

| Health Camp ID 6578 | Patient ID 489652  | Primary Key 4896526578  |
|---------------------|--------------------|-------------------------|

Creating this primary key was helpful in combining information and creating additional time features; meaningful data was spread among several csv files. 
![]( https://github.com/AChezick/Capstone2/blob/main/images/images2/primary_key2.png )


### The need for Feature Engineering 

Training the model with only the five anonymized features results in very poor performance.

![]( https://github.com/AChezick/Capstone2/blob/main/images/non_feature_all_models.png )

The two anonymized features that had the highest feature weights were Var1 , Var5. Interestingly however, most of the patients had a zero-value for these two features. Without knowing what 'var1' is, and given that only a few thousand patients had non-zero values, I decided not to drop or edit these features for modeling purposes. There is simply not enough context to apply domain knowledge for the features Var1 - Var5.

![]( https://github.com/AChezick/Capstone2/blob/main/images/images2/var1_5_Zero_.png ) 

Thus, feature engineering was instrumental in improving the model. 

#### One Hot Encoding & Imputing 

#### Categorical Features and Imputation 

The categorical features include: City & Job Type. The binary categorical features pertained to if a patient shared their health fair attendance online through Twitter, LinkedIn, FaceBook, or were an online follower of MedCamp.

Most patients had many missing values for Job Type and other numerical features (discussed later). To avoid co-lineraity, I imputed 9999.0 for the missing values in the Job column.

--- 

| Var1 | Var2 | Var3 | Var4 | Var5 | y_target | Camp Start Date - Registration Date | Registration Date - First Interaction | Camp Start Date - First Interaction | Camp End Date - Registration Date | Camp Length | 1036 | 1216 | 1217 | 1352 | 1704 | 1729 | 2517 | 2662 | 23384 | Second | Third | B | C | D | E | F | G | 2100 | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 | 7.0 | 8.0 | 9.0 | 10.0 | 11.0 | 12.0 | 13.0 | 14.0 | 9999.0 | 1 | 2 | 3 | 4 |
|------|------|------|------|------|----------|-------------------------------------|---------------------------------------|-------------------------------------|-----------------------------------|-------------|------|------|------|------|------|------|------|------|-------|--------|-------|---|---|---|---|---|---|------|-----|-----|-----|-----|-----|-----|-----|-----|------|------|------|------|------|--------|---|---|---|---|
| 4.0  | 0.0  | 0.0  | 0.0  | 2.0  | 1.0      | -25.0                               | 278                                   | 253                                 | 34                                | 59          | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 1     | 0      | 1     | 0 | 0 | 0 | 0 | 0 | 1 | 1    | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0    | 0    | 0    | 0    | 0    | 1      | 0 | 0 | 0 | 0 |
| 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0      | -24.0                               | 99                                    | 75                                  | 161                               | 185         | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 1     | 0      | 0     | 0 | 0 | 0 | 0 | 1 | 0 | 1    | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0    | 0    | 0    | 0    | 0    | 1      | 0 | 0 | 0 | 0 |
| 4.0  | 0.0  | 0.0  | 0.0  | 2.0  | 0.0      | -60.0                               | 355                                   | 295                                 | 711                               | 771         | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 1     | 0      | 0     | 0 | 0 | 0 | 0 | 1 | 0 | 1    | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0    | 0    | 0    | 0    | 0    | 1      | 0 | 0 | 0 | 0 |

--- 

Nearly 23,500 patients were missing the Camp Location. However, I was able to use the primary key to link a patient with a camp. Then, using sets, I was able to confirm that each camp ID is only associated with a certain city value by checking for unions and intersections among CampIDs, PatientIDs and Camp Length that were spread among the csv files. Thus, I was able to backtrack and impute missing city values for each patient which did improve prediction scores.


#### Numerical Features and Imputation 

The numerical features provided by MedCamp were missing for most paitents. For example Age, Income and Education Score had less than 2,000 value each. Given that, 94% of the patients were missing all three values; imputing average values onto the other 35,000 patients for any numerical feature would be meaningless and create colinearity. 

![]( https://github.com/AChezick/Capstone2/blob/main/images/images2/attendance_catgegorical.png ) 


### Features from Dates

I used the primary key to track the unique patient events and consolidate important information into csv that could be used for training and testing. 

![]( https://github.com/AChezick/Capstone2/blob/main/images/images2/primary_key.png ) 

The following features were created:

|   | Feature Name (Days)                   |
|---|---------------------------------------|
|   | Registration Date - First Interaction |
|   | Camp Start Date - Registration Date   |
|   | Camp Start Date - First Interaction   |
|   | Camp End Date - Registration Date     |
|   | Camp Length                           | 


---


### Modeling 
Given the goal is to ensure all patients have an individualized health experience , there has to be specific supplies. Having accurate predictions means we can be confident in having the correct supplies and accomplishing  the goal for improving health through individualized interventions. 

--- 

#### Results after creating features, one-hot encoding, scaling 

![]( https://github.com/AChezick/Capstone2/blob/main/images/roc_all%20models.png ) 

As shown above all models achelved a similar ROC score. However, when we take the number of false negatives and false positives into consideration going with the XG Boost model is the best choice. 

![]( https://github.com/AChezick/Capstone2/blob/main/images/images2/pde_campLength.png ) 

The Date features ended up improving scores for all models. Additionally, for all but some iterations of Random Forests, the date/times features would show among top feature importances. 

![]( https://github.com/AChezick/Capstone2/blob/main/images/feature_importance_XG2.png ) 

---
 

### 

## Post-Hoc

The global attendance rate was 20%. The training and validation attendance rate was 27%. However, 5/10 camp locations had a attendance rate between 32.2% and 33.8%. The highest attendance rate was just over 70%. The high level of variance helps to explain why adjusting to the exact glabal attendance rate, when dealing with class imbalance, casued the models to perform worse than with the standard balanced class option. However, models did perform best with a slight weighting of classes at .4 for attends and .6 for non-attends.


#### There was much diversity BOTH within & among Health Camp attendance rates as it pertains to:
##### 1. The size of the Health Camp. 
##### 2. Among groups of the same size
##### 3. Camp Location
##### 4. Among different camps at the Same Location

![]( https://github.com/AChezick/Capstone2/blob/main/images/images2/scatter_camp2.png )

#### There is a correlation and outlier among Health Camp Attendance Rates:

![]( https://github.com/AChezick/Capstone2/blob/main/images/images2/test_size_attendsRate_city.png ) 

#### Models Specifics: Disagreement on 'Which Patients will attend' 

I created a new dataframe which contains the prediction and probability results for three of the models used in this project. 
y_target_SUM is the total 'Score' or sum of predicted attendance (0 or 1) among all models and y_target. Top value = 4 
Y_count_allModels is the the sum of all predicted values for attendance (0 or 1) among the three models being analyzed here Top value = 3

---

|   | Unnamed: 0 | Var1 | Var2 | Var3 | Var4 | Var5 | Camp Start Date - Registration Date | Registration Date - First Interaction | Camp Start Date - First Interaction | Camp End Date - Registration Date | Camp Length | Second | Third | A | C | D | E | F | G | 2100 | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 | 7.0 | 8.0 | 9.0 | 10.0 | 11.0 | 12.0 | 13.0 | 14.0 | 9999.0 | 1 | 2 | 3 | 4 | 1036 | 1216 | 1217 | 1352 | 1704 | 1729 | 2517 | 2662 | 23384 | Patient_ID | prediction | Proba      | y_target | proba_kNN | prediction_kNN | proba_sVC          | prediction_sVC | proba_xg   | prediction_xg | Y_count_allModels | Y_target_SUM |
|---|------------|------|------|------|------|------|-------------------------------------|---------------------------------------|-------------------------------------|-----------------------------------|-------------|--------|-------|---|---|---|---|---|---|------|-----|-----|-----|-----|-----|-----|-----|-----|------|------|------|------|------|--------|---|---|---|---|------|------|------|------|------|------|------|------|-------|------------|------------|------------|----------|-----------|----------------|--------------------|----------------|------------|---------------|-------------------|--------------|
| 0 | 0          | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | -119.0                              | 14                                    | -105                                | 66                                | 185         | 0      | 0     | 0 | 0 | 0 | 0 | 1 | 0 | 1    | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0    | 0    | 0    | 0    | 0    | 1      | 0 | 0 | 0 | 0 | 0    | 0    | 0    | 0    | 0    | 0    | 1    | 0    | 0     | 514789     | 0.0        | 0.2775604  | 0.0      | 0.0       | 0.0            | 0.1561240540625182 | 0.0            | 0.28253844 | 0.0           | 0.0               | 0.0          |
| 1 | 1          | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | -410.0                              | 559                                   | 149                                 | 361                               | 771         | 0      | 0     | 0 | 0 | 0 | 0 | 1 | 0 | 1    | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0    | 0    | 0    | 0    | 0    | 1      | 0 | 0 | 0 | 0 | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 1     | 508149     | 0.0        | 0.24640098 | 0.0      | 0.1       | 0.0            | 0.1557039638243477 | 0.0            | 0.23559786 | 0.0           | 0.0               | 0.0          |
| 2 | 2          | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | -76.0                               | 262                                   | 186                                 | 113                               | 189         | 0      | 0     | 0 | 0 | 0 | 0 | 1 | 0 | 1    | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0    | 0    | 0    | 0    | 0    | 0      | 0 | 0 | 0 | 0 | 0    | 0    | 0    | 0    | 0    | 0    | 1    | 0    | 0     | 492650     | 0.0        | 0.33918345 | 0.0      | 0.2       | 0.0            | 0.1573992543873874 | 0.0            | 0.34879157 | 0.0           | 0.0               | 0.0          |
| 3 | 3          | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 53.0                                | 107                                   | 160                                 | 57                                | 4           | 1      | 0     | 1 | 0 | 0 | 0 | 0 | 0 | 1    | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0    | 0    | 0    | 0    | 0    | 1      | 0 | 0 | 0 | 0 | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 1     | 511274     | 0.0        | 0.4557019  | 0.0      | 0.3       | 0.0            | 0.1542467704410321 | 0.0            | 0.43797377 | 0.0           | 0.0               | 0.0          |
| 4 | 4          | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 19.0                                | 11                                    | 30                                  | 58                                | 39          | 0      | 1     | 0 | 0 | 0 | 0 | 0 | 1 | 1    | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0    | 0    | 0    | 0    | 0    | 1      | 0 | 0 | 0 | 0 | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 1     | 520795     | 1.0        | 0.5163712  | 0.0      | 0.4       | 0.0            | 0.5654859255622244 | 1.0            | 0.4760046  | 0.0           | 1.0               | 1.0          |

--- 

Upon closer examination there is disagreement among the models for which patients will attend a health event. It might be possible to gain useful insight be examining interesting patients: those which models agreed, disagreed, False Positives, False Negatives, etc.  


Below are plots showing the following for each Patient: 
* A. The probability each model assigned to a patient (y-axis) 
* B. If the patient actually attended (shown by color)  
* C. The overall score group for that patient (shown by the respective column the dot appears in) 
 
![]( https://github.com/AChezick/Capstone2/blob/main/images/images2/ytarget_svcprobas.png ) 
![]( https://github.com/AChezick/Capstone2/blob/main/images/images2/ytarget_knnprobas.png ) 
![]( https://github.com/AChezick/Capstone2/blob/main/images/images2/ytarget_xgprobas.png ) 
 

#### Experimentation with Tensorflow & Keras
Since this data is not descriptive black-box models are OK to use. Optimization of a neural network may produce good results. I used tensorflow and keras and was able to achieve similar results to other models with minimal training. 

![]( https://github.com/AChezick/Capstone2/blob/main/images/images2/BasicNN.png )

However, I am confident that these scores can improve by using a grid search and other optimization techniques. 

![]( https://github.com/AChezick/Capstone2/blob/main/images/images2/BasicNN_loss.png ) 

---

# ~ AB Testing in /src

*Under construction

My plan is to conduct a mock analysis of 'model predictions' had they been actually implemented. Essentially, 'What if' MedCamp used previous camp data to train models for each camp individually and sequentially?


### Which model (SVC, Logistic Regression, KNN) would perform best as a bandit ?!? 

Steps in Experiment:

1. Put camps in-order by end date
2A. Remove overlap (if  for Camp D , Camps A,B & C, end  before Camp D, the patient data from Camp A,B &C would be used to train the [SVC,Logistic Regression, KNN] to predict Camp D’s patient attendance). 
2B. However, if Camp C starts before D but does not end before D starts Camp C ‘s results can’t be used to train the bandits [SVC,Logistic Regression, KNN]. 
3. Append model results to data frame

Steps for modified Thompson Sampling:

I modified the traditional Thompson Sampling in favor for a numerical solution. Rather than explictely using the[ beta ](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html), a small penality ( - 0.5%) was imposed if a bandit is chosen AND the Beta was greater than the actual rate for that model. In a different experiment using different bandits this method acheived a modest improvement over using the exact Beta. I plan on conducting the same experiment for this data set. Results comming soon.

Below is a graph showing what Beta was chosen and how it compares to that bandit's current win rate

![]( https://github.com/AChezick/Capstone2/blob/main/images/images2/beta_vs_winrate.png )


## Results

Win = Correct prediction of a random patient's attendance for that camp.

Initial results are mixed, with some camps having improved prediction rates and some being worse. Three rounds of results are shown in the table below.

When I separated each camp and had the models predict patient attendance for just that camp, each model generally performed better than when I had used more data and trained them all at once. My next step will be to see how scores align with other features: 
--Camp Location, Camp Length etc. As indicated in the post-hoc above, there was much variation among camp attendance rates and this may result in poor performance. 
--The data may have been 'pulled' away from a better prediction vector by too much diversity and not enough data among the diversity to create a normal distribution.

|    | camp_ID | Win Rate SVC | Win Rate KNN | Win Rate Logistic Regression | Camp Size (Number of Patients) | knn2  | svc2  | log2  | knn3  | svc3  | log3  |
|----|---------|--------------|--------------|------------------------------|--------------------------------|-------|-------|-------|-------|-------|-------|
| 0  | 6578    | 0.345        | 0.336        | 0.362                        | 2835                           | 0.361 | 0.348 | 0.349 | 0.366 | 0.335 | 0.338 |
| 1  | 6532    | 0.874        | 0.861        | 0.863                        | 1991                           | 0.5   | 0.5   | 0.667 | 0.5   | 0.6   | 0.5   |
| 2  | 6543    | 0.88         | 0.877        | 0.852                        | 6541                           | 0.883 | 0.865 | 0.857 | 0.885 | 0.864 | 0.856 |
| 3  | 6580    | 0.935        | 0.925        | 0.913                        | 3515                           | 0.921 | 0.934 | 0.925 | 0.92  | 0.934 | 0.919 |
| 4  | 6570    | 0.92         | 0.918        | 0.927                        | 3562                           | 0.913 | 0.925 | 0.926 | 0.932 | 0.915 | 0.908 |
| 5  | 6542    | 0.836        | 0.81         | 0.866                        | 2366                           | 0.85  | 0.828 | 0.857 | 0.845 | 0.842 | 0.853 |
| 6  | 6571    | 0.92         | 0.892        | 0.899                        | 2084                           | 0.911 | 0.897 | 0.908 | 0.902 | 0.906 | 0.908 |
| 7  | 6527    | 0.379        | 0.344        | 0.428                        | 4142                           | 0.413 | 0.41  | 0.384 | 0.406 | 0.32  | 0.43  |
| 8  | 6526    | 0.954        | 0.969        | 0.959                        | 3807                           | 0.97  | 0.966 | 0.944 | 0.965 | 0.965 | 0.958 |
| 9  | 6539    | 0.883        | 0.861        | 0.871                        | 1990                           | 0.333 | 0.75  | 0.5   | 0.5   | 0.5   | 0.667 |
| 10 | 6528    | 0.306        | 0.305        | 0.243                        | 1742                           | 0.5   | 0.667 | 0.5   | 0.5   | 0.667 | 0.5   |
| 11 | 6555    | 0.646        | 0.61         | 0.489                        | 1736                           | 0.5   | 0.333 | 0.5   | 0.5   | 0.333 | 0.5   |
| 12 | 6541    | 0.409        | 0.382        | 0.429                        | 1545                           | 0.5   | 0.333 | 0.5   | 0.5   | 0.333 | 0.5   |
| 13 | 6523    | 0.277        | 0.258        | 0.315                        | 2082                           | 0.303 | 0.3   | 0.29  | 0.29  | 0.309 | 0.264 |
| 14 | 6538    | 0.841        | 0.837        | 0.845                        | 3952                           | 0.839 | 0.843 | 0.843 | 0.839 | 0.845 | 0.84  |
| 15 | 6549    | 0.682        | 0.719        | 0.687                        | 1833                           | 0.75  | 0.667 | 0.5   | 0.5   | 0.667 | 0.75  |
| 16 | 6586    | 0.766        | 0.789        | 0.675                        | 2622                           | 0.789 | 0.723 | 0.769 | 0.798 | 0.745 | 0.681 |
| 17 | 6554    | 0.843        | 0.859        | 0.846                        | 2301                           | 0.844 | 0.857 | 0.848 | 0.839 | 0.863 | 0.835 |
| 18 | 6529    | 0.495        | 0.573        | 0.577                        | 3821                           | 0.581 | 0.556 | 0.51  | 0.564 | 0.566 | 0.574 |
| 19 | 6540    | 0.888        | 0.901        | 0.906                        | 1424                           | 0.5   | 0.5   | 0.4   | 0.667 | 0.333 | 0.333 |
| 20 | 6534    | 0.305        | 0.296        | 0.283                        | 3595                           | 0.303 | 0.287 | 0.285 | 0.28  | 0.303 | 0.306 |
| 21 | 6535    | 0.871        | 0.831        | 0.884                        | 1880                           | 0.5   | 0.75  | 0.667 | 0.8   | 0.5   | 0.5   |
| 22 | 6561    | 0.542        | 0.685        | 0.592                        | 198                            | 0.5   | 0.5   | 0.333 | 0.5   | 0.5   | 0.4   |
| 23 | 6585    | 0.735        | 0.787        | 0.752                        | 1396                           | 0.5   | 0.6   | 0.5   | 0.5   | 0.5   | 0.667 |
| 24 | 6536    | 0.565        | 0.57         | 0.495                        | 2035                           | 0.573 | 0.493 | 0.557 | 0.554 | 0.549 | 0.564 |
| 25 | 6562    | 0.963        | 0.959        | 0.957                        | 2336                           | 0.956 | 0.957 | 0.968 | 0.968 | 0.959 | 0.946 |
| 26 | 6537    | 0.881        | 0.864        | 0.877                        | 3857                           | 0.878 | 0.876 | 0.869 | 0.883 | 0.838 | 0.883 |
| 27 | 6581    | 0.943        | 0.935        | 0.938                        | 1483                           | 0.667 | 0.5   | 0.75  | 0.667 | 0.5   | 0.75  |
| 28 | 6524    | 0.655        | 0.538        | 0.667                        | 147                            | 0.667 | 0.667 | 0.333 | 0.5   | 0.667 | 0.5   |
| 29 | 6587    | 0.462        | 0.581        | 0.556                        | 77                             | 0.531 | 0.536 | 0.542 | 0.308 | 0.6   | 0.556 |
| 30 | 6557    | 0.35         | 0.3          | 0.294                        | 50                             | 0.375 | 0.227 | 0.368 | 0.125 | 0.316 | 0.367 |
| 31 | 6546    | 0.931        | 0.967        | 0.936                        | 401                            | 0.957 | 0.904 | 0.966 | 0.951 | 0.946 | 0.949 |
| 32 | 6569    | 0.374        | 0.357        | 0.296                        | 175                            | 0.5   | 0.2   | 0.5   | 0.333 | 0.5   | 0.25  |
| 33 | 6564    | 0.915        | 0.884        | 0.885                        | 512                            | 0.88  | 0.912 | 0.901 | 0.893 | 0.911 | 0.886 |
| 34 | 6575    | 0.36         | 0.32         | 0.533                        | 88                             | 0.467 | 0.387 | 0.441 | 0.413 | 0.333 | 0.475 |
| 35 | 6552    | 0.125        | 0.514        | 0.548                        | 80                             | 0.429 | 0.529 | 0.5   | 0.381 | 0.538 | 0.519 |
| 36 | 6558    | 0.583        | 0.526        | 0.5                          | 42                             | 0.72  | 0.462 | 0.273 | 0.588 | 0.455 | 0.571 |

## Next Steps


Question to ponder:
- Is there conflicting data within and among camp locations? 
--As in patient A has values 1,1,1 for characteristics ABC, and does NOT attend and patient B has 1,1,1 for characteristics ABC, and DOES attend? Thinking about this as permutations there would be 2^N different ways for each binary outcome.

- Clustering patients might reveal trends and predictors 
- I would like to see if multi-class models can predict how many models successfully predicted a patients attendance
-- As in if 4 models were successful (success = 4) is there a meta pattern that could be learned?

