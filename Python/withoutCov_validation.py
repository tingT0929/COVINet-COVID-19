# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 02:56:50 2020

@author: Yuting Zhang
"""

import os
import pandas as pd
import math
import numpy as np
from keras.models import Model
from keras.layers import Input, Activation, Dense, LSTM, Dropout, concatenate
from keras.layers.recurrent import GRU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

def mean_relative_error(y_true, y_pred):
    temp1 = np.abs(y_true[:, 0] - y_pred[:, 0]) / y_true[:, 0]
    temp2 = np.abs(y_true[:, 1] - y_pred[:, 1]) / y_true[:, 1]
    temp1 = temp1[~np.isinf(temp1)]
    temp2 = temp2[~np.isinf(temp2)]
    relative_error1 = np.nanmean(temp1, axis=0)
    relative_error2 = np.nanmean(temp2, axis=0)
    relative_error = np.nanmean([relative_error1, relative_error2])
    return relative_error1, relative_error2, relative_error

path = '。/data'
os.chdir(path)

np.random.seed(7)
##############################
# load data
##################################
dataset = pd.read_csv('USA_data_county_smooth_no.csv')
dataset = dataset.sort_values(by = ['date'], axis = 0)
backupDF = dataset.reset_index(drop = True)
dataset.reset_index(inplace = True, drop = True)
    
look_back = 7
remain_day = 30 #7
save_path = path + '/result/withoutCov'

dataset_or = dataset.drop(['StateCounty', 'date', 'county', 'state'], axis = 1)
dataset_or_con = dataset_or.iloc[0, :look_back*2:2].values.tolist()
dataset_or_con = dataset_or_con + dataset.cum_confirm.values.tolist()

dataset_or_dead = dataset_or.iloc[0, 1:look_back*2:2].values.tolist()
dataset_or_dead = dataset_or_dead + dataset.cum_dead.values.tolist()

dataset_or = np.array([dataset_or_con, dataset_or_dead])
dataset_or = np.transpose(dataset_or, (1, 0))

tempTime = dataset['date'].unique()
tempTime.sort()
valDF = dataset[dataset['date'].isin(tempTime[-remain_day:])]
dataset = dataset[~dataset['date'].isin(tempTime[-remain_day:])]

valDF.drop(['StateCounty', 'date', 'county', 'state'], axis = 1, inplace = True)
dataset.drop(['StateCounty', 'date', 'county', 'state'], axis = 1, inplace = True)
    
colname = list(dataset.columns.values)
trainDF = dataset.reset_index(drop = True)
valDF.reset_index(drop = True, inplace = True)

##################################
#create scaler for covid-19
##################################
dataDF = pd.read_csv('us-counties-smooth.csv')

dataDF['county'] = dataDF['county'].map(lambda x: x[:-4]+'City' if x[-4:] == 'city' else x)
dataDF.loc[dataDF['county'] == 'New York City', 'county'] = 'New York'
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

StateCounty = dataDF.apply(lambda x: x['state'] + ' ' + x['county'], axis = 1)
dataDF.insert(0, 'StateCounty', StateCounty)

dataDF = dataDF.sort_values(by = ['date'], axis = 0)
dataDF = dataDF[dataDF.StateCounty.isin(backupDF.StateCounty)]
dataDF.drop(['StateCounty', 'date', 'county', 'state', 'fips'], axis = 1, inplace = True)
dataDF = dataDF.append({'cases':0, 'deaths':0}, ignore_index = True)
dataDF.reset_index(inplace = True, drop = True)

##################################
# scale the train data
##############################
scaler = MinMaxScaler(feature_range=(0, 1))
dataDF = scaler.fit_transform(dataDF)

for i in range(look_back + 1):
    trainDF.iloc[:, 2*i:2*i+2] = scaler.transform(trainDF.iloc[:, 2*i:2*i+2])
    valDF.iloc[:, 2*i:2*i+2] = scaler.transform(valDF.iloc[:, 2*i:2*i+2])
trainDF = pd.DataFrame(trainDF, columns = colname)
train = trainDF.values

train_x, train_y = train[:, :-2], train[:, -2:]
train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])

valDF = pd.DataFrame(valDF, columns = colname)
val = valDF.values

val_x, val_y = val[:, :-2], val[:, -2:]
val_x = val_x.reshape(val_x.shape[0], 1, val_x.shape[1])

weight_file = save_path + '/weights/lstm_no_pred.h5'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
if not os.path.isdir(save_path+'/weights'):
    os.makedirs(save_path+'/weights')

############################
######## network
############################
epochs = 100
mainInputs = Input(shape=[train_x.shape[1], train_x.shape[2]], name = 'main_input')
x1 = GRU(50, input_shape = (train_x.shape[1], train_x.shape[2]), name = 'gru_1')(mainInputs)
x1 = Dropout(0.05, name = 'dropout_1')(x1)
x2 = LSTM(50, input_shape = [train_x.shape[1], train_x.shape[2]], name = 'lstm_1')(mainInputs)
x2 = Dropout(0.2, name = 'dropout_2')(x2)

x3 = concatenate([x1, x2], axis = 1, name = 'concatenate_1')
    
x4 = Dense(train_y.shape[1], name = 'dense_1')(x3)
x4 = Activation("relu", name = 'activation_1')(x4)
model = Model(mainInputs, x4)

##############################
## train model 
##############################
model.compile(loss='mean_squared_error', optimizer="adam", metrics=["accuracy"])
plot_model(model, to_file = save_path + '/model.png')#, show_shapes=True)
early_stopping = EarlyStopping(patience=40, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.3, patience=10, min_lr=0.00001, verbose=1)
checkpointer = ModelCheckpoint(filepath = weight_file, verbose=1, 
                               monitor='val_loss', mode='auto', save_best_only=True)


history = model.fit(train_x, train_y, epochs = epochs, batch_size = 32,
                        validation_data = (val_x, val_y), 
                        verbose = 1, shuffle = False,
                        callbacks = [reduce_lr, checkpointer, early_stopping])
    
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'val')
plt.legend()
plt.show()

##############################
## load model
##############################
model.load_weights(weight_file)

#########################
############ predict
#########################
## top 10 county
os.chdir(path)
countyNum = 10

dataset = pd.read_csv('USA_data_county_smooth_no.csv')
dataset = dataset.sort_values(by = ['date'], axis = 0)

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

np.random.seed(7)
for i in range(len(maxCountyList)):
    stateCountyName = maxCountyList[i]
    dataset_2 = datasetDF.sort_values(by = ['date'], axis = 0)
    dataset_2.reset_index(inplace = True, drop = True)
    dataset_2 = dataset_2[dataset_2['StateCounty'] == stateCountyName]
    
    tempTime = dataset_2['date'].unique()
    tempTime.sort()
    valDF = dataset_2[dataset_2['date'].isin(tempTime[-remain_day:])]
    dataset_2 = dataset_2[~dataset_2['date'].isin(tempTime[-remain_day:])]
    
    valDF.drop(['StateCounty', 'date', 'county', 'state'], axis = 1, inplace = True)
    dataset_2.drop(['StateCounty', 'date', 'county', 'state'], axis = 1, inplace = True)
    
    colname = list(dataset_2.columns.values)
    trainDF = dataset_2.reset_index(drop = True)
    valDF.reset_index(drop = True, inplace = True)
    
    for i in range(look_back + 1):
        trainDF.iloc[:, 2*i:2*i+2] = scaler.transform(trainDF.iloc[:, 2*i:2*i+2])
        valDF.iloc[:, 2*i:2*i+2] = scaler.transform(valDF.iloc[:, 2*i:2*i+2])
    trainDF = pd.DataFrame(trainDF, columns = colname)
    train = trainDF.values
    
    train_x, train_y = train[:, :-2], train[:, -2:]
    train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
    
    valDF = pd.DataFrame(valDF, columns = colname)
    val = valDF.values
    
    val_x, val_y = val[:, :-2], val[:, -2:]
    val_x = val_x.reshape(val_x.shape[0], 1, val_x.shape[1])
    
    ########## validation
    trainPredict = model.predict(train_x)#预测训练集
    ## RMSE
    trainScore = math.sqrt(mean_squared_error(train_y, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    
    trainPredict = scaler.inverse_transform(trainPredict)
    train_Y = scaler.inverse_transform(train_y)
    
    trainPredict = trainPredict.astype(np.int32)
    
    trainScore = math.sqrt(mean_squared_error(train_Y, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
        
    trainCase, trainDeath, trainMRE = mean_relative_error(train_Y, trainPredict)
    print('Train Mean Relative Error: %.2f' % (trainMRE))
        
    #### val
    valPredict = model.predict(val_x)#预测训练集
    ## RMSE
    valScore = math.sqrt(mean_squared_error(val_y, valPredict))
    print('val Score: %.2f RMSE' % (valScore))
    
    valPredict = scaler.inverse_transform(valPredict)
    val_Y = scaler.inverse_transform(val_y)
    
    valPredict = valPredict.astype(np.int32)
    
    valScore = math.sqrt(mean_squared_error(val_Y, valPredict))
    print('val Score: %.2f RMSE' % (valScore))
        
    valCase, valDeath, valMRE = mean_relative_error(val_Y, valPredict)
    print('val Mean Relative Error: %.2f' % (valMRE))
    
    valRMSEList.append(valScore)
    valCaseList.append(valCase)
    valDeathList.append(valDeath)
    valMREList.append(valMRE)

testMetric = np.array([maxCountyList, valRMSEList, valCaseList, valDeathList, valMREList])
testMetric = np.transpose(testMetric, (1, 0))
testMetric = pd.DataFrame(testMetric, 
                          columns = ['StateCounty', 'RMSES', 'Case MRE', 'Death MRE', 'MRE'])
testMetric.reset_index(drop = True, inplace = True)
testMetric.to_csv(save_path + '/valMetric_{}.csv'.format(remain_day), index = False)
