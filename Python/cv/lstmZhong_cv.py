# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 15:38:26 2021

lstm (zhong)

@author: Yuting zhang
"""

import os
import pandas as pd
import math
import numpy as np
from dbfread import DBF
from sklearn.metrics import mean_squared_error
from keras.models import model_from_json
from sklearn.model_selection import KFold
from functions import *

import warnings
warnings.filterwarnings('ignore')

path = './data'
os.chdir(path)

look_back = 7
np.random.seed(7)
##############################
# load training data
##############################
dataset = pd.read_csv('USA_data_county_smooth.csv')

##############################
# create scaler for covid-19 & factor
##############################
dataDF = pd.read_csv('us-counties-smooth.csv')
factorDF = pd.read_csv('USA_factor_county.csv')
latLongTable = DBF('lat_long.dbf')
latLongDF = pd.DataFrame(iter(latLongTable))
latLongDF.drop(['STATE', 'CWA', 'TIME_ZONE', 'FE_AREA'], axis = 1, inplace = True)
latLongDF.drop_duplicates(subset = ['FIPS'], inplace = True)
#latLongDF.to_csv('temp.csv', index = False)
latLongDF['FIPS'] = latLongDF['FIPS'].astype(np.int64)
latLongDF.rename(columns = {'FIPS': 'fips'}, inplace = True)

scalerCOVID, scalerFactor, scalerLatLong, \
        factorColname_or, latLongDF = get_scaler(dataset, dataDF,
                                                 factorDF, latLongDF,
                                                 look_back)
factorColname = factorColname_or.copy()

##################################
## CV 10 fold
resultRoot = '/public/home/jiangyukang/kanny/covid_cov_DL/data'
resultCol = ['cv time', 'from',
             'total RMSE', 'total case MRE', 'total death MRE', 'total MRE',
             'val RMSE','val case MRE', 'val death MRE', 'val MRE',
             ]
cvResult = pd.DataFrame(columns = resultCol)

remain_day = [7, 30]
cvNum = 10
StateCounty_series = dataset.StateCounty.drop_duplicates()
StateCounty_remain = StateCounty_series.tolist()
kf = KFold(n_splits=cvNum, random_state=2021, shuffle=True)
countyIndex = kf.split(StateCounty_remain)
cvTimes = 0
cvCounty = {'train':{}, 'test':{}}

for trainCountyIndex, testCountyIndex in countyIndex:
    cvTimes += 1
    trainCounty = StateCounty_series.iloc[trainCountyIndex].tolist()
    testCounty = StateCounty_series.iloc[testCountyIndex].tolist()
    
    cvCounty['train'][cvTimes]  = trainCounty.copy()
    cvCounty['test'][cvTimes]  = testCounty.copy()
    trainMainDF, trainAuxDF, trainYDF, \
        valMainDF, valAuxDF, valYDF = get_cv_train_data_v2(dataset, latLongDF,
                                                           scalerCOVID, scalerFactor,
                                                           scalerLatLong, look_back,
                                                           trainCounty)
    isTrain = 1

    save_path = path + '/result/cvCompare/lstm_zhong/cov_cv{}'.format(cvTimes)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(save_path + '/data'):
        os.makedirs(save_path + '/data')
    if not os.path.isdir(save_path + '/weights'):
        os.makedirs(save_path + '/weights')
    
    os.chdir(path)    
    weight_file = save_path + '/weights/lstm_county.h5'
    
    model, train_main_x, train_y, \
        val_main_x, val_y = training_lstm_noCov_model(trainMainDF, trainYDF,
                                                     valMainDF, valYDF,
                                                     save_path, weight_file,
                                                     isTrain = isTrain)

    model.load_weights(weight_file)
    json_string = model.to_json()
    open(save_path + '/weights/lstm_county.json', 'w').write(json_string)

    #########################
    ############ predict train
    #########################
    trainPredict = model.predict(train_main_x)#???????????????
    
    # inverse scaler
    trainPredict = scalerCOVID.inverse_transform(trainPredict)
    train_Y = scalerCOVID.inverse_transform(train_y)
    trainPredict = trainPredict.astype(np.int32)
    totalTrainScore = math.sqrt(mean_squared_error(train_Y, trainPredict))
    print('Train Score: %.2f RMSE' % (totalTrainScore))
    totalTrainCase, totalTrainDeath, totalTrainMRE = mean_relative_error(train_Y, trainPredict)
    print('Train Mean Relative Error: %.2f' % (totalTrainMRE))
    
    ### val
    valPredict = model.predict(val_main_x)#??????val???
    
    # inverse scaler
    valPredict = scalerCOVID.inverse_transform(valPredict)
    val_Y = scalerCOVID.inverse_transform(val_y)
    valPredict = valPredict.astype(np.int32)
    totalvalScore = math.sqrt(mean_squared_error(val_Y, valPredict))
    print('val Score: %.2f RMSE' % (totalvalScore))
    totalvalCase, totalvalDeath, totalvalMRE = mean_relative_error(val_Y, valPredict)
    print('val Mean Relative Error: %.2f' % (totalvalMRE))
    
    cvResult = cvResult.append(pd.DataFrame([[cvTimes, 'train', totalTrainScore, 
                                   totalTrainCase, totalTrainDeath, 
                                   totalTrainMRE, totalvalScore,
                                   totalvalCase, totalvalDeath, 
                                   totalvalMRE]], columns = resultCol),
                                ignore_index = True)
    
    #########################
    ############ predict test
    #########################
    for d in remain_day:
        # calculate latest 7 and 30 day
        testMainDF, testAuxDF, testYDF, \
            testValMainDF, testValAuxDF, testValYDF = get_cv_train_data_v2(dataset, latLongDF,
                                                               scalerCOVID, scalerFactor,
                                                               scalerLatLong, look_back,
                                                               testCounty, d)
        test_main_x, test_aux_x, test_y, \
            test_val_main_x, test_val_aux_x, test_val_y = get_cv_test_data(testMainDF, testAuxDF, testYDF, 
                                                                           testValMainDF, testValAuxDF, testValYDF, 
                                                                           factorColname)
        
        testPredict = model.predict(test_main_x)#???????????????
        # inverse scaler
        testPredict = scalerCOVID.inverse_transform(testPredict)
        test_y = scalerCOVID.inverse_transform(test_y)
        testPredict = testPredict.astype(np.int32)
        totalTestScore = math.sqrt(mean_squared_error(test_y, testPredict))
        print('Test Score before inverse: %.2f RMSE, remain %d day' % (totalTestScore, d))
        totalTestCase, totalTestDeath, totalTestMRE = mean_relative_error(test_y, testPredict)
        print('Test Mean Relative Error: %.2f, remain %d day' % (totalTestMRE, d))
        
        ### val
        test_valPredict = model.predict(test_val_main_x)#???????????????
        # inverse scaler
        test_valPredict = scalerCOVID.inverse_transform(test_valPredict)
        test_val_Y = scalerCOVID.inverse_transform(test_val_y)
        test_valPredict = test_valPredict.astype(np.int32)
        totaltest_valScore = math.sqrt(mean_squared_error(test_val_Y, test_valPredict))
        print('val Score: %.2f RMSE' % (totaltest_valScore))
        totaltest_valCase, totaltest_valDeath, totaltest_valMRE = mean_relative_error(test_val_Y, test_valPredict)
        print('val Mean Relative Error: %.2f' % (totaltest_valMRE))
    
        cvResult = cvResult.append(pd.DataFrame([[cvTimes, 'test remain {}'.format(d), 
                                                  totalTestScore, totalTestCase, totalTestDeath,  
                                                  totalTestMRE, totaltest_valScore, totaltest_valCase, 
                                                  totaltest_valDeath, totaltest_valMRE]], columns = resultCol),
                                    ignore_index = True)


### predict all county with 10fold
for d in remain_day:
    trainMainDF, trainAuxDF, trainYDF, \
        valMainDF, valAuxDF, valYDF, dataset_or = get_train_data(dataset, latLongDF,
                                                                 scalerCOVID, scalerFactor,
                                                                 scalerLatLong, look_back, d)
    train_main_x, train_aux_x, train_y, \
            val_main_x, val_aux_x, val_y = get_cv_test_data(trainMainDF, trainAuxDF, trainYDF, 
                                                            valMainDF, valAuxDF, valYDF, 
                                                            factorColname)
    
    trainCase = [0]*train_y.shape[0]
    trainDeath = [0]*train_y.shape[0]
    valCase = [0]*val_y.shape[0]
    valDeath = [0]*val_y.shape[0]
    
    for random_seed in range(1, cvTimes+1):
        weight_path = path + '/result/cvCompare/lstm_zhong/cov_cv{}'.format(random_seed)
        model = model_from_json(open(weight_path + '/weights/lstm_county.json').read())
        model.load_weights(weight_path + '/weights/lstm_county.h5')
        
        trainPredict = model.predict(train_main_x)#???????????????
        valPredict = model.predict(val_main_x)#???????????????
        
        trainCase += trainPredict[:,0]
        trainDeath += trainPredict[:,1]
        valCase += valPredict[:,0]
        valDeath += valPredict[:,1]
    
    
    trainPredict = np.array([trainCase, trainDeath]) / cvNum
    trainPredict = np.transpose(trainPredict, (1, 0))
    # inverse scaler
    trainPredict = scalerCOVID.inverse_transform(trainPredict)
    train_Y = scalerCOVID.inverse_transform(train_y)
    trainPredict = trainPredict.astype(np.int32)
    totalTrainScore = math.sqrt(mean_squared_error(train_Y, trainPredict))
    print('Mean train Score: %.2f RMSE' % (totalTrainScore))
    totalTrainCase, totalTrainDeath, totalTrainMRE = mean_relative_error(train_Y, trainPredict)
    print('Mean train Mean Relative Error: %.2f' % (totalTrainMRE))
    
    ### val
    valPredict = np.array([valCase, valDeath]) / cvNum
    valPredict = np.transpose(valPredict, (1, 0))
    # inverse scaler
    valPredict = scalerCOVID.inverse_transform(valPredict)
    val_Y = scalerCOVID.inverse_transform(val_y)
    valPredict = valPredict.astype(np.int32)
    totalvalScore = math.sqrt(mean_squared_error(val_Y, valPredict))
    print('val Score: %.2f RMSE' % (totalvalScore))
    totalvalCase, totalvalDeath, totalvalMRE = mean_relative_error(val_Y, valPredict)
    print('val Mean Relative Error: %.2f' % (totalvalMRE)) 
    
    cvResult = cvResult.append(pd.DataFrame([['total', 'test remain {}'.format(d), totalTrainScore, 
                                   totalTrainCase, totalTrainDeath, 
                                   totalTrainMRE, totalvalScore,
                                   totalvalCase, totalvalDeath, 
                                   totalvalMRE]], columns = resultCol),
                                ignore_index = True)
    
cvResult.reset_index(drop = True, inplace = True)
cvResult.to_csv(path + '/result/cvCompare/lstm_zhong/cvMetric.csv', index = False)
