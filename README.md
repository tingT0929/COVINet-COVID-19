# COVINet

- Developed deep learning by incorporating non-sequencing predictors to project the sequence outcomes
- Emphasized the covariates (COV) interpreted (I) by the deep learning network (Net). 
- To compare the performance of our model, we used the deep learning model for LSTM, and GRU separately and random forest as alternative tools.   

## 1. Data 
### 1) Abstract
Our data source consists of cumulative confirmed cases and deaths from late January 2020  to March 23, 2023 which were collected from New York Times. The contributing factors related to adverse health were compiled from the County Health Rankings and Roadmaps program official website. These data were available in their offical websites. This enabled us to predict COVID-19 in the United States using the interpretable variables associated to COVID-19 .

### 2) Availability
The data to reproduce our results are available. [Dropbox Data]:(https://www.dropbox.com/scl/fo/e5161r096y0mi7pex5m5h/h?rlkey=3ponpcl52jkfiutij2dwj23ep&dl=0) [BaiduNetdisk Data]: (https://pan.baidu.com/s/1gvXsjrOMEnDafBSwSK7K-w?pwd=bo75) password：bo75

### 3) Permissions
The data were orignially collected by the authors.

----
## 2. Code
### 1)  Abstract
The codes incorported all the scripts to reproduce the analysis in the paper. 

### 2) Reporducibility
- The random forest results by runing R scripts.
- The results of LSTM only, GRU only, LSTM+GRU without covariates, and LSTM+GRU (our proposed model) by runing Python script data_pro.py first to generate data and running lstm_cov_factor_server.py to run the model.
----
