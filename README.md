# Capstone2

The goal of my project is to help MedCamp -a healthfair provider company – reduce wasteful spending AND maintain quality experiences of attendies by accuretly predcting who will and will not attend a healthfare. Specifically reducing false positives and negaitves. 

--- 

## Data 
This project, Healthcare Analytics, came from [Kaggle](https://www.kaggle.com/vin1234/janatahack-healthcare-analytics?select=Train) 

#### Kaggle Description 


train.zip contains 6 different csv files apart from the data dictionary as described below:

**Health_Camp_Detail.csv** – File containing HealthCampId, CampStartDate, CampEndDate and Category details of each camp.

**Train.csv** – File containing registration details for all the test camps. This includes PatientID, HealthCampID, RegistrationDate and a few anonymized variables as on registration date.

---
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-0lax"></th>
    <th class="tg-0lax">Patient_ID</th>
    <th class="tg-0lax">Health_Camp_ID</th>
    <th class="tg-0lax">Registration_Date</th>
    <th class="tg-0lax">Var1</th>
    <th class="tg-0lax">Var2</th>
    <th class="tg-0lax">Var3</th>
    <th class="tg-0lax">Var4</th>
    <th class="tg-0lax">Var5</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">489652</td>
    <td class="tg-0lax">6578</td>
    <td class="tg-0lax">10-Sep-05</td>
    <td class="tg-0lax">4</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">2</td>
  </tr>
  <tr>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">507246</td>
    <td class="tg-0lax">6578</td>
    <td class="tg-0lax">18-Aug-05</td>
    <td class="tg-0lax">45</td>
    <td class="tg-0lax">5</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">7</td>
  </tr>
  <tr>
    <td class="tg-0lax">2</td>
    <td class="tg-0lax">523729</td>
    <td class="tg-0lax">6534</td>
    <td class="tg-0lax">29-Apr-06</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
  </tr>
  <tr>
    <td class="tg-0lax">3</td>
    <td class="tg-0lax">524931</td>
    <td class="tg-0lax">6535</td>
    <td class="tg-0lax">07-Feb-04</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
  </tr>
  <tr>
    <td class="tg-0lax">4</td>
    <td class="tg-0lax">521364</td>
    <td class="tg-0lax">6529</td>
    <td class="tg-0lax">28-Feb-06</td>
    <td class="tg-0lax">15</td>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">0</td>
    <td class="tg-0lax">7</td>
  </tr>
</tbody>
</table>

--- 


**Patient_Profile.csv** – This file contains Patient profile details like PatientID, OnlineFollower, Social media details, Income, Education, Age, FirstInteractionDate, CityType and EmployerCategory
      

**First_Health_Camp_Attended.csv** – This file contains details about people who attended health camp of first format. This includes Donation (amount) & Health_Score of the person.

**Second_Health_Camp_Attended.csv** - This file contains details about people who attended health camp of second format. This includes Health_Score of the person.

**Third_Health_Camp_Attended.csv** - This file contains details about people who attended health camp of third format. This includes Numberofstallvisited & LastStallVisitedNumber.

**Test.csv** – File containing registration details for all the camps done after 1st April 2006. This includes PatientID, HealthCampID, RegistrationDate and a few anonymized variables as on registration date. Participant should make predictions for these patient camp combinations


### EDA


---


### Modeling 

[[16088  2418]]
 [[ 9349  7394]]


---


### Discussion 


---


### Next Steps
