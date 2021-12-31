# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:24:25 2020

@author: Yuting Zhang
"""

import os
import pandas as pd
import numpy as np
from dbfread import DBF
from functions import *

import warnings
warnings.filterwarnings('ignore')

path = './data'
os.chdir(path)

look_back = 7
noNewYork = 1
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
#latLongDF.to_csv('temp.csv', index = False)
latLongDF['FIPS'] = latLongDF['FIPS'].astype(np.int64)
latLongDF.rename(columns = {'FIPS': 'fips'}, inplace = True)

if noNewYork == 1:
    dataset = dataset[dataset['StateCounty'] != 'New York New York']
    dataDF = dataDF[dataDF['county'] != 'New York City']

scalerCOVID, scalerFactor, scalerLatLong, \
        factorColname_or, latLongDF = get_scaler(dataset, dataDF,
                                                 factorDF, latLongDF,
                                                 look_back)

##################################
# scale the train data
##############################
trainMainDF, trainAuxDF, trainYDF, \
        valMainDF, valAuxDF, valYDF, dataset_or = get_train_data(dataset, latLongDF,
                                                                 scalerCOVID, scalerFactor,
                                                                 scalerLatLong, look_back)

##############################
## Driving alone to work; Traffic volume; Income inequality
##############################
stayCol = [25, 33, 42]
stayCol.sort()
countyNum = 10
isTrain = 1 

if noNewYork == 1:
    save_path = path + '/result/noNY/cov_p{}_factor{}'.format(countyNum, len(stayCol))
else:
    save_path = path + '/result/withNY/cov_p{}_factor{}'.format(countyNum, len(stayCol))
ny_path = path + '/result/withNY/cov_p{}_factor{}'.format(countyNum, len(stayCol))

if not os.path.isdir(save_path):
    os.makedirs(save_path)
if not os.path.isdir(save_path + '/weights'):
    os.makedirs(save_path + '/weights')
    
os.chdir(path)
factorColname = factorColname_or.copy()
for i in stayCol:
    factorColname.remove(factorColname_or[i])

weight_file = save_path + '/weights/lstm_county_cov_predict_{}.h5'.format(len(stayCol))
ny_weight_file = ny_path + '/weights/lstm_county_cov_predict_{}.h5'.format(len(stayCol))

##############################
## train model 
##############################
model, train_main_x, train_aux_x, train_y,\
    val_main_x, val_aux_x, val_y = training_model(trainMainDF, trainAuxDF, trainYDF,
                                                  valMainDF, valAuxDF, valYDF,
                                                  factorColname, save_path, weight_file,
                                                  ny_weight_file = ny_weight_file, 
                                                  noNewYork = noNewYork,
                                                  isTrain = isTrain)

##############################
## save model
##############################
model.load_weights(save_path + '/weights/lstm_county_cov_predict_{}.h5'.format(len(stayCol)))
json_string = model.to_json()
open(save_path + '/weights/lstm_county_cov_{}.json'.format(len(stayCol)), 'w').write(json_string)
