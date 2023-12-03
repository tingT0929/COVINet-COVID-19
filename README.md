# COVINet: A deep learning-based and interpretable prediction model for the county-wise trajectories of COVID-19 in the United States

- Developed deep learning by incorporating non-sequencing predictors to project the sequence outcomes
- Emphasized the covariates (COV) interpreted (I) by the deep learning network (Net). 
- To compare the performance of our model, we used the deep learning model for LSTM, and GRU separately and random forest as alternative tools.   

------
## 1. Data 
### 1) Abstract
Our data source consists of cumulative confirmed cases and deaths from late January 2020  to March 23, 2023 which were collected from [New York Times](https://github.com/nytimes/covid-19-data). The contributing factors related to adverse health were compiled from the [County Health Rankings and Roadmaps program official website](https://www.countyhealthrankings.org/). These data were available in their offical websites. This enabled us to predict COVID-19 in the United States using the interpretable variables associated to COVID-19 .

### 2) Availability
The data to reproduce our results are available. 
- [Dropbox](https://www.dropbox.com/scl/fo/e5161r096y0mi7pex5m5h/h?rlkey=3ponpcl52jkfiutij2dwj23ep&dl=0) 
- [BaiduNetdisk](https://pan.baidu.com/s/1gvXsjrOMEnDafBSwSK7K-w?pwd=bo75) (password：bo75)

### 3) Permissions
The data were orignially collected by the authors.

----
## 2. Code
### 1)  Abstract
The codes incorported all the scripts to reproduce the analysis in the paper. 

### 2) Reporducibility
- Variable importance is determined by executing R scripts in the case of random forest results.
- To obtain results for LSTM only, GRU only, LSTM+GRU without covariates, and LSTM+GRU (our proposed model), the process involves first generating data using the Python script `data_pro.py` and then running the model through `lstm_cov_factor_server.py`.

----

## 3. Results

## 1) Variable importance

The variable importance results in the random forest are modeled based on the cumulative confirmed cases and deaths count, with the importance metric as the Gini index. The corresponding importance metrics have excluded the cumulative confirmed and death counts from the first 14 days.

|       Covariates                               | Cumulative confirmed cases |
|------------------------------------------------------|------------------|
| **Population**                                         | 211833894914515  |
| **Some college**                                    | 152316590417946  |
| **Average traffic volume per meter of major roadways in the county** | 61141975648333 |
| **Severe housing problems**                          | 54518584078595   |
| **Access to exercise opportunities**           | 31813747393460 |
| **Average daily density of fine particulate matter in micrograms per cubic meter (PM2.5)**                                | 22742795917040   |

|        Covariates                          | Cumulative Deaths   |
|------------------------------------------------------|------------------|
| **Some college**                                    | 30405146567 |
| **Population**                                         | 25830917809 |
| **Average traffic volume per meter of major roadways in the county** | 11646886505 |
| **Severe housing problems**                         | 5497518119 |
| **Average daily density of fine particulate matter in micrograms per cubic meter (PM2.5)**                                | 4585140703 |
| **Access to exercise opportunities**                 | 4115907117 |



## 2) MSEs and MREs

Comparison of the performance of COVINet and 10 CDC models in predicting the disease dynamics at the county level using the MAE and MRE as the evaluation metrics. The results are reported for the top 10 states and all states in the US for a 7-day prediction horizon.

| |Top Ten States | Top Ten States| All States|All States |
|--------------------|----------------------------|---------------------------------------------|--------------------|---------------------|
| **Method**                  | **MAE**$_7$      | **MRE**${}_7$    | **MAE**${}_7$ | **MRE**${}_7$ |
| COVINet(Our Proposed Model)                    | **125.00**                            | **0.0039**                         | **54.45**   | 0.0097             |
| UMass-MechBayes           | 163.05                                      | 0.0058                                  | 58.00              | 0.0079             |
| COVIDhub CDC-ensemble   | 167.88                                      | 0.0060                                  | 58.10              | **0.0077**    |
| LANL-GrowthRate          | 173.55                                      | 0.0063                                  | 62.72              | 0.0080             |
| MOBS-GLEAM COVID      | 179.64                                      | 0.0065                                  | 66.37              | 0.0083             |
| COVIDhub-baseline  | 186.20                                      | 0.0067                                  | 71.64              | 0.0100             |
| IowaStateLW-STEM | 187.25                                      | 0.0065                                  | 73.46              | 0.0082             |
| UT-Mobility        | 196.75                                      | 0.0072                                  | 72.33              | 0.0083             |
| CU-select         | 221.26                                      | 0.0074                                  | 82.07              | 0.0096             |
| CU-nochange            | 221.78                                      | 0.0075                                  | 82.18              | 0.0096             |
| JHU-IDD-CovidSP    | 409.73                                      | 0.0157                                  | 132.71             | 0.0216             |
