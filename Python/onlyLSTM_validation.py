# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:26:25 2021

@author: Yuting Zhang
"""

import os
import pandas as pd
import math
import numpy as np
from dbfread import DBF
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from functions import *

import warnings
warnings.filterwarnings('ignore')

path = './data'
os.chdir(path)

look_back = 7
remain_day = 30 # 7
np.random.seed(7)

##############################
# load data
##################################
dataset = pd.read_csv('USA_data_county_smooth.csv')

##################################
#create scaler for covid-19 & factor
##################################
dataDF = pd.read_csv('us-counties-smooth.csv')
factorDF = pd.read_csv('USA_factor_county.csv')
latLongTable = DBF('lat_long.dbf')
latLongDF = pd.DataFrame(iter(latLongTable))
latLongDF.drop(['STATE', 'CWA', 'TIME_ZONE', 'FE_AREA'], axis = 1, inplace = True)
latLongDF['FIPS'] = latLongDF['FIPS'].astype(np.int64)
latLongDF.rename(columns = {'FIPS': 'fips'}, inplace = True)
    
scalerCOVID, scalerFactor, scalerLatLong, \
        factorColname_or, latLongDF = get_scaler(dataset, dataDF,
                                                 factorDF, latLongDF,
                                                 look_back, remain_day)

##################################
# scale the train data
##############################
trainMainDF, trainAuxDF, trainYDF, \
        valMainDF, valAuxDF, valYDF, dataset_or = get_train_data(dataset, latLongDF,
                                                                 scalerCOVID, scalerFactor,
                                                                 scalerLatLong, look_back,
                                                                 remain_day)

##############################
## Driving alone to work; Traffic volume; Income inequality
##############################
stayCol = [25, 33, 42]
stayCol.sort()
countyNum = 10
isTrain = 1

save_path = path + '/result/lstm'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
if not os.path.isdir(save_path + '/weights'):
    os.makedirs(save_path + '/weights')
os.chdir(path)

factorColname = factorColname_or.copy()
for i in stayCol:
    factorColname.remove(factorColname_or[i])
    
weight_file = save_path + '/weights/lstm_county_cov.h5'

##############################
## train model 
##############################
model, train_main_x, train_aux_x, train_y, \
    val_main_x, val_aux_x, val_y = training_lstm_model(trainMainDF, trainAuxDF, trainYDF,
                                                           valMainDF, valAuxDF, valYDF,
                                                           factorColname, save_path, weight_file,
                                                           isTrain = isTrain)

##############################
## load model
##############################
model.load_weights(weight_file)

json_string = model.to_json()
open(save_path + '/weights/lstm_county_cov.json', 'w').write(json_string)

#########################
############ predict
#########################
## top 10 county
datasetDF = pd.read_csv('USA_data_county_smooth.csv')
datasetDF = datasetDF.sort_values(by = ['date'], axis = 0)
datasetDF.reset_index(inplace = True, drop = True)

tempTime = datasetDF['date'].unique()
tempTime.sort()

maxCountyList = datasetDF[datasetDF['date'] == tempTime[-1]]
maxCountyList = maxCountyList.sort_values(by = ['cum_confirm'], axis = 0, )
maxCountyList = maxCountyList.StateCounty.tolist()[-countyNum:]

valRMSEList = []
valCaseList = []
valDeathList = []
valMREList = []

for i in range(len(maxCountyList)):
    stateCountyName = maxCountyList[i]
    dataset = datasetDF[datasetDF['StateCounty'] == stateCountyName]
    tempTime = dataset['date'].unique()
    tempTime.sort()
    
    dataDF = pd.read_csv('us-counties-smooth.csv')
    dataDF['county'] = dataDF['county'].map(lambda x: x[:-4]+'City' if x[-4:] == 'city' else x)
    dataDF.loc[dataDF['county'] == 'Fairbanks North Star Borough', 'county'] = 'Fairbanks North Star'
    dataDF.loc[dataDF['county'] == 'Ketchikan Gateway Borough', 'county'] = 'Ketchikan Gateway'
    dataDF.loc[dataDF['county'] == 'Kenai Peninsula Borough', 'county'] = 'Kenai Peninsula'
    dataDF.loc[dataDF['county'] == 'Doña Ana', 'county'] = 'Dona Ana'
    dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Juneau City and Borough'), 'county'] = 'Juneau'
    dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Matanuska-Susitna Borough'), 'county'] = 'Matanuska-Susitna'
    dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Bethel Census Area'), 'county'] = 'Bethel'
    dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Kodiak Island Borough'), 'county'] = 'Kodiak Island'
    dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Nome Census Area'), 'county'] = 'Nome'
    dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Petersburg Borough'), 'county'] = 'Petersburg'
    dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Prince of Wales-Hyder Census Area'), 'county'] = 'Prince of Wales-Hyder'
    dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Southeast Fairbanks Census Area'), 'county'] = 'Southeast Fairbanks'
    dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Yukon-Koyukuk Census Area'), 'county'] = 'Yukon-Koyukuk'
    
    StateCountyAll = dataDF.apply(lambda x: x['state'] + ' ' + x['county'], axis = 1)
    dataDF.insert(0, 'StateCounty', StateCountyAll)
    if stateCountyName == 'New York New York':
        dataDF = dataDF[dataDF['StateCounty'] == 'New York New York City']
    else:
        dataDF = dataDF[dataDF['StateCounty'] == stateCountyName]
    dataDF.drop(['StateCounty'], axis = 1, inplace = True)
    
    factorDF = pd.read_csv('USA_factor_county.csv')
    latLongTable = DBF('lat_long.dbf')
    latLongDF = pd.DataFrame(iter(latLongTable))
    latLongDF.drop(['STATE', 'CWA', 'TIME_ZONE', 'FE_AREA'], axis = 1, inplace = True)
    latLongDF['FIPS'] = latLongDF['FIPS'].astype(np.int64)
    latLongDF.rename(columns = {'FIPS': 'fips'}, inplace = True)
    

    scalerCOVID_single, scalerFactor_single, scalerLatLong_single, \
        factorColname_or, latLongDF = get_scaler(dataset, dataDF,
                                                 factorDF, latLongDF,
                                                 look_back, remain_day)
    trainMainDF, trainAuxDF, trainYDF,\
            valMainDF, valAuxDF, valYDF, dataset_or = get_train_data(dataset, latLongDF,
                                                                     scalerCOVID_single, scalerFactor_single,
                                                                     scalerLatLong_single, look_back, 
                                                                     remain_day)
    
    trainAuxDF.drop(factorColname, axis = 1, inplace = True) 
    valAuxDF.drop(factorColname, axis = 1, inplace = True) 
    
    train_main_x = trainMainDF.values
    train_aux_x = trainAuxDF.values
    train_y = trainYDF.values
    
    val_main_x = valMainDF.values
    val_aux_x = valAuxDF.values
    val_y = valYDF.values
    
    train_main_x = train_main_x.reshape(train_main_x.shape[0], 1, train_main_x.shape[1])
    val_main_x = val_main_x.reshape(val_main_x.shape[0], 1, val_main_x.shape[1])
    
    ########## validation
    trainPredict = model.predict([train_main_x, train_aux_x])
    ## RMSE
    trainScore = math.sqrt(mean_squared_error(train_y, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    
    testPredict = model.predict([val_main_x, val_aux_x])
    ## val RMSE
    testScore = math.sqrt(mean_squared_error(val_y, testPredict))
    print('Val RMSE: %.2f RMSE' % (testScore))
    
    ### 
    trainPredict = scalerCOVID_single.inverse_transform(trainPredict)
    train_Y = scalerCOVID_single.inverse_transform(train_y)
    testPredict = scalerCOVID_single.inverse_transform(testPredict)    
    val_Y = scalerCOVID_single.inverse_transform(val_y) 
    
    trainPredict = trainPredict.astype(np.int32)
    testPredict = testPredict.astype(np.int32)
    
    trainScore = math.sqrt(mean_squared_error(train_Y, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    
    testScore = math.sqrt(mean_squared_error(val_Y, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))
    valRMSEList.append(testScore)
    
    valCase, valDeath, valMRE = mean_relative_error(val_Y, testPredict)
    valCaseList.append(valCase)
    valDeathList.append(valDeath)
    valMREList.append(valMRE)
    
    tempTime = pd.to_datetime(tempTime)
    tempTime = tempTime.tolist()
    
    tempTime = [min(tempTime)+timedelta(-(look_back - j)) for j in range(look_back)] + tempTime
    tempTime = [d.date() for d in tempTime]
    
testMetric = np.array([maxCountyList, valRMSEList, valCaseList, valDeathList, valMREList])
testMetric = np.transpose(testMetric, (1, 0))
testMetric = pd.DataFrame(testMetric, 
                          columns = ['StateCounty', 'RMSES', 'Case MRE', 'Death MRE', 'MRE'])
testMetric.reset_index(drop = True, inplace = True)
testMetric.to_csv(save_path + '/lstmMetric_{}.csv'.format(remain_day), index = False)