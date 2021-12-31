# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:24:25 2020

@author: Yuting Zhang
"""

import os
import pandas as pd
import math
import numpy as np
from dbfread import DBF
from datetime import timedelta
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.models import model_from_json
from functions import *

import warnings
warnings.filterwarnings('ignore')

path = './data'
os.chdir(path)

testLen = 30
look_back = 7
stayCol = [25, 33, 42]
stayCol.sort()
countyNum = 10

save_path = path + '/result/cov_p{}_factor{}'.format(countyNum, len(stayCol))
if not os.path.isdir(save_path):
    os.makedirs(save_path)
if not os.path.isdir(save_path + '/data'):
    os.makedirs(save_path + '/data')

##############################
## load model
##############################
weight_path = path + '/result/noNY/cov_p{}_factor{}/weights'.format(countyNum, len(stayCol))
model = model_from_json(open(weight_path + '/lstm_county_cov_{}.json'.format(len(stayCol))).read())
model.load_weights(weight_path + '/lstm_county_cov_predict_{}.h5'.format(len(stayCol)))

weight_Dense_1,bias_dense_1 = model_all.get_layer('dense_1').get_weights()
stayColWeight = pd.DataFrame(weight_Dense_1[-5:], columns = ['cases', 'deaths'])

##############################
## load data
##############################
dataset = pd.read_csv('USA_data_county_smooth.csv')

#####################    
datasetDF = dataset.sort_values(by = ['date'], axis = 0)
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

# top 10 county
np.random.seed(7)
for i in range(len(maxCountyList)):
    stateCountyName = maxCountyList[i]
    dataset = datasetDF[datasetDF['StateCounty'] == stateCountyName]
    tempTime = dataset['date'].unique()
    tempTime.sort()
        
    # create scaler for covid-19 & factor
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
                                                 look_back)
    factorColname = factorColname_or.copy()
    stayFactorColname = []
    for i in stayCol:
        stayFactorColname.append(factorColname_or[i])
        factorColname.remove(factorColname_or[i])

    ## train data
    trainMainDF, trainAuxDF, trainYDF,\
            valMainDF, valAuxDF, valYDF, dataset_or = get_train_data(dataset, latLongDF,
                                                                     scalerCOVID_single, scalerFactor_single,
                                                                     scalerLatLong_single, look_back)
    
    trainAuxDF.drop(factorColname, axis = 1, inplace = True) 

    train_main_x = trainMainDF.values
    train_aux_x = trainAuxDF.values
    train_y = trainYDF.values
    train_main_x = train_main_x.reshape(train_main_x.shape[0], 1, train_main_x.shape[1])
    
    ## val data
    valAuxDF.drop(factorColname, axis = 1, inplace = True) 

    val_main_x = valMainDF.values
    val_aux_x = valAuxDF.values
    val_y = valYDF.values
    val_main_x = val_main_x.reshape(val_main_x.shape[0], 1, val_main_x.shape[1])

    ########## predict
    testLen = 30
    testx_con = [0.]*(testLen+look_back+1)
    testx_con[0:look_back] = train_y[-(look_back+1):, 0].tolist()
    
    testx_dead = [0.]*(testLen+look_back+1)
    testx_dead[0:look_back] = train_y[-(look_back+1):, 1].tolist()
    
    testx_factor = train_aux_x[0,]
    testx_factor = np.reshape(testx_factor, (1, train_aux_x.shape[1]))
    
    testx = np.array([testx_con, testx_dead])
    testx = np.transpose(testx, (1, 0))
    
    testPredict_con = [0]*(testLen+1)
    testPredict_dead = [0]*(testLen+1)
    
    testxxx = []
    for j in range(testLen+1):
        testxx = testx[j:(j+look_back)]
        testxx = np.reshape(testxx, (1, 1, look_back * 2))
        testy = model.predict([testxx, testx_factor])
        
        testxxx.append(testxx.tolist()[0][0])
        tempCase = testy.tolist()[0][0]
        tempDeath = testy.tolist()[0][1]
        if j >= 1 and tempCase < testx[look_back+j-1][0]:
            tempCase = testx[look_back+j-1][0]
        if j >= 1 and tempDeath < testx[look_back+j-1][1]:
            tempDeath = testx[look_back+j-1][1]
        
        if j >= 1:
            testx[look_back+j] = [tempCase, tempDeath]
        testPredict_con[j] = tempCase
        testPredict_dead[j] = tempDeath
    
    testPredict = np.array([testPredict_con, testPredict_dead])
    testPredict = np.transpose(testPredict, (1, 0))
    testxxx = np.array(testxxx)
    
    trainPredict = model.predict([train_main_x, train_aux_x])
    valPredict = model.predict([val_main_x, val_aux_x])
        
    #train RMSE
    trainScore = math.sqrt(mean_squared_error(train_y, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    
    trainPredict = scalerCOVID_single.inverse_transform(trainPredict)
    train_Y = scalerCOVID_single.inverse_transform(train_y)
    valPredict = scalerCOVID_single.inverse_transform(valPredict)
    val_Y = scalerCOVID_single.inverse_transform(val_y)
    testPredict = scalerCOVID_single.inverse_transform(testPredict) 
    
    trainPredict = trainPredict.astype(np.int32)
    valPredict = valPredict.astype(np.int32)
    testPredict = testPredict.astype(np.int32)
    
    trainScore = math.sqrt(mean_squared_error(train_Y, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    
    trainCase, trainDeath, trainMRE = mean_relative_error(train_Y, trainPredict)
    print('Train Mean Relative Error: %.2f' % (trainMRE))
    
    valScore = math.sqrt(mean_squared_error(val_Y, valPredict))
    print('val Score: %.2f RMSE' % (valScore))
    
    valCase, valDeath, valMRE = mean_relative_error(val_Y, valPredict)
    print('val Mean Relative Error: %.2f' % (valMRE))
    valRMSEList.append(valScore)
    valCaseList.append(valCase)
    valDeathList.append(valDeath)
    valMREList.append(valMRE)
    
    ##################
    ## plot 
    ##################
    tempTime = pd.to_datetime(tempTime)
    tempTime = tempTime.tolist()
    tempTime = [min(tempTime)+timedelta(-(look_back - j)) for j in range(look_back)] + tempTime
    tempTime = tempTime + [max(tempTime)+timedelta(j+1) for j in range(testLen)]
    tempTime = [d.date() for d in tempTime]
    
    #######
    # case
    datasetOrPlot = np.reshape(np.array([None]*len(tempTime)), (len(tempTime), 1))
    datasetOrPlot[:, :] = np.nan
    datasetOrPlot[:len(dataset_or), :] = dataset_or[:, 0].reshape((dataset_or.shape[0], 1))
    
    trainPredictPlot = np.reshape(np.array([None]*len(tempTime)), (len(tempTime), 1))
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict[:, 0].reshape((trainPredict.shape[0], 1))
    
    testPredictPlot = np.reshape(np.array([None]*len(tempTime)), (len(tempTime), 1))
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(dataset_or)-1:(len(dataset_or)+testLen), :] = testPredict[:, 0].reshape((testPredict.shape[0], 1))
    
    plt.plot(tempTime, datasetOrPlot, label='True')
    plt.plot(tempTime, trainPredictPlot, label='Train Predict')
    plt.plot(tempTime, testPredictPlot, label='Predict')
    plt.title(stateCountyName + ' Cases', fontsize = 14)
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(save_path + '/' + stateCountyName + '_countyCov_confirm.png')
    plt.show()
    
    ## death
    datasetOrPlot = np.reshape(np.array([None]*len(tempTime)), (len(tempTime), 1))
    datasetOrPlot[:, :] = np.nan
    datasetOrPlot[:len(dataset_or), :] = dataset_or[:, 1].reshape((dataset_or.shape[0], 1))
    
    trainPredictPlot = np.reshape(np.array([None]*len(tempTime)), (len(tempTime), 1))
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict[:, 1].reshape((trainPredict.shape[0], 1))
    
    testPredictPlot = np.reshape(np.array([None]*len(tempTime)), (len(tempTime), 1))
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(dataset_or)-1:(len(dataset_or)+testLen), :] = testPredict[:, 1].reshape((testPredict.shape[0], 1))
    
    plt.plot(tempTime, datasetOrPlot, label='True')
    plt.plot(tempTime, trainPredictPlot, label='Train Predict')
    plt.plot(tempTime, testPredictPlot, label='Predict')
    plt.title(stateCountyName + ' Deaths', fontsize = 14)
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(save_path + '/' + stateCountyName + '_countyCov_dead.png')
    plt.show()
    
    ######### to csv
    ######### case
    dataset_or_case = [np.nan] * len(tempTime)
    dataset_or_case[:len(dataset_or)] = dataset_or[:, 0].tolist()
    trainPredict_case = [np.nan] * len(tempTime)
    trainPredict_case[look_back:len(trainPredict)+look_back] = trainPredict[:, 0].tolist()
    testPredict_case = [np.nan] * len(tempTime)
    testPredict_case[len(dataset_or)-1:(len(dataset_or)+testLen)] = testPredict[:, 0].tolist()
    
    #### death
    dataset_or_death = [np.nan] * len(tempTime)
    dataset_or_death[:len(dataset_or)] = dataset_or[:, 1].tolist()
    trainPredict_death = [np.nan] * len(tempTime)
    trainPredict_death[look_back:len(trainPredict)+look_back] = trainPredict[:, 1].tolist()
    testPredict_death = [np.nan] * len(tempTime)
    testPredict_death[len(dataset_or)-1:(len(dataset_or)+testLen)] = testPredict[:, 1].tolist()
    
    predDF = np.array([tempTime, dataset_or_case, dataset_or_death,
                       trainPredict_case, trainPredict_death,
                       testPredict_case, testPredict_death])
    predDF = np.transpose(predDF, (1, 0))
    predDF = pd.DataFrame(predDF, 
                          columns = ['date', 'true_case', 'true_death',
                                     'train_case', 'train_death',
                                     'pred_case', 'pred_death'])
    predDF.to_csv(save_path + '/data/' + stateCountyName + '_pred.csv', index = False)
    
valMetric = np.array([maxCountyList, valRMSEList, valCaseList, valDeathList, valMREList])
valMetric = np.transpose(valMetric, (1, 0))
valMetric = pd.DataFrame(valMetric,
                         columns = ['StateCounty', 'RMSES', 'Case MRE', 'Death MRE', 'MRE'])
valMetric.reset_index(drop = True, inplace = True)
valMetric.to_csv(save_path + '/valMetric.csv', index = False)
