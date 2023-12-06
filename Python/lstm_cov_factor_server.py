# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 22:26:52 2020

@author: zhang
"""
#import keras
#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
#
import os
import pandas as pd
import math
from dbfread import DBF
import numpy as np
from keras.models import Model
from keras.layers import Input, Activation, Dense, LSTM, Dropout, concatenate
from keras.layers import GRU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import datetime
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from function_for_lstmCOV import *
import argparse
from keras.models import load_model


parser = argparse.ArgumentParser()
parser.add_argument('--file_dataDF', type = str, default= 'us-2023.csv')
parser.add_argument('--file_dataset', type = str, default= 'data-14-new.csv')
parser.add_argument('--model', type = str, default='model')
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--target', type = int, default = 1) # 0 for cases and 1 for deaths
parser.add_argument('--limit_date', type = str, default = '')

args = parser.parse_args()
file_dataDF = args.file_dataDF
file_dataset = args.file_dataset 
model_type = args.model
seed = args.seed
target = args.target
limit_date = args.limit_date
path = '~./data'
os.chdir(path)

latLongTable = DBF('lat_long.dbf')
latLongDF = pd.DataFrame(iter(latLongTable))
latLongDF.drop(['STATE', 'CWA', 'TIME_ZONE', 'FE_AREA'], axis = 1, inplace = True)
latLongDF['FIPS'] = latLongDF['FIPS'].astype(np.int64)
latLongDF.rename(columns = {'FIPS': 'fips'}, inplace = True)

look_back = 14
remain_day = 7 * 8
testLen = 40
stayCol = ['Average Daily PM2.5', 'Average Traffic Volume per Meter of Major Roadways', '% Severe Housing Problems', 'LON', 'LAT']
top10states = ['Florida', 'Louisiana', 'Connecticut', 'California', 'Michigan', 'Pennsylvania', 'Illinois', 'Massachusetts', 'New Jersey', 'New York']

dataset = pd.read_csv(file_dataset)
if len(limit_date):
    dataset = dataset[dataset['date'] <= limit_date]
print(dataset.columns)


#####################
#create scaler for covid-19 & factor
dataDF = pd.read_csv(file_dataDF)
try:
    dataset['date'] = [datetime.datetime.strptime(dataset['date'][i], '%Y/%m/%d') for i in range(len(dataset))]
    dataDF['date'] = [datetime.datetime.strptime(dataDF['date'][i], '%Y/%m/%d') for i in range(len(dataDF))]
except:
    pass

if target:
    dataset.drop(['confirm_14', 'confirm_13', 'confirm_12','confirm_11','confirm_10', 'confirm_9', 'confirm_8', 'confirm_7', 'confirm_6','confirm_5','confirm_4', 'confirm_3', 'confirm_2','confirm_1', 'cum_confirm'], axis = 1, inplace = True)
    dataDF.drop(['cases'], axis = 1, inplace = True)
else:
    dataset.drop(['dead_14', 'dead_13', 'dead_12', 'dead_11', 'dead_10', 'dead_9', 'dead_8', 'dead_7', 'dead_6','dead_5','dead_4', 'dead_3', 'dead_2','dead_1', 'cum_dead'], axis = 1, inplace = True)
    dataDF.drop(['deaths'], axis = 1, inplace = True)
    

factorDF = pd.read_csv('USA_factor_county.csv')
print("data loaded...")

dataset, dataDF = data_process(dataset, dataDF)

top_10_county, scalerCOVID, scalerFactor, scalerLatLong, factorColname_or, latLongDF = get_scaler(dataset, dataDF, factorDF, latLongDF, look_back, top10states, target, remain_day)
print("top_10_county:", top_10_county)
top_10_county = ['Florida Miami-Dade', 'Louisiana Jefferson', 'Connecticut Fairfield', 'California Los Angeles', 'Michigan Wayne', 'Pennsylvania Philadelphia', 'Illinois Cook', 'Massachusetts Middlesex', 'New Jersey Bergen', 'New York New York City']
print("scaler loaded...")

trainMainDF, trainAuxDF, trainYDF, valMainDF, valAuxDF, valYDF, dataset_or, state, state_train, state_test = get_train_data(dataset, latLongDF,
                                                                 scalerCOVID, scalerFactor,
                                                                 scalerLatLong, look_back,
                                                                 target, remain_day, seed = seed)

datasetDF = dataset.sort_values(by = ['date'], axis = 0)
datasetDF.reset_index(inplace = True, drop = True)

tempTime = datasetDF['date'].unique()
tempTime = np.sort(tempTime)


save_path = path + '/result/cov_factor_0621_single'

model = factor_model(trainMainDF, trainAuxDF, trainYDF, valMainDF, valAuxDF, valYDF, 
                                                             dataset_or, save_path, 
                                                             scalerCOVID, scalerFactor,
                                                             look_back, remain_day, state, state_train, state_test,
                                                             stayCol, model_type = "model", seed = seed)

weight_Dense_1,bias_dense_1 = model.get_layer('dense_1').get_weights()
print("weight", weight_Dense_1[-5:, :])
stayColWeight = weight_Dense_1[-1]

stayFactorColname = stayCol
changeFactor = [0.5, 1, 2, 3, 4]


for start_index in range(6):
    date_list = []
    pre30_ = []
    true_ = []
    for sc_ in top_10_county:
        print(sc_)
        trainMainDF, trainAuxDF, trainYDF, date, stateName = get_county_data(dataset, latLongDF, stayCol, state, scalerCOVID, scalerFactor, scalerLatLong, sc_, look_back,remain_day)
        for cfi in range(3):
            for cf in changeFactor:
                pre30, true = predict_county(model, save_path, trainMainDF, trainAuxDF, trainYDF, remain_day, scalerCOVID, state, stateName, look_back, start_index, cfi, cf)
                pre30_.append(pre30)
                true_.append(true[:,0])
                date_list.append(date)

    date_list = np.array(date_list)
    np.save('/public/home/tianting/lstm/data/changefactor/date_' + str(start_index) + '_' + str(seed) + '_' + str(target) +'.npy', date_list)
    pre30_ = np.array(pre30_)
    np.save('/public/home/tianting/lstm/data/changefactor/pre30_'+ str(start_index) + '_' + str(seed) + '_'+ str(target) +  '.npy', pre30_)
    true_ = np.array(true_)
    np.save('/public/home/tianting/lstm/data/changefactor/true_' + str(start_index) + '_' + str(seed) + '_'+ str(target) +  '.npy', true_)
    np.save('/public/home/tianting/lstm/data/changefactor/cfilist_' + str(sd) + '_' + str(seed) + '_'+ str(target) +  '.npy', cfi_list)

