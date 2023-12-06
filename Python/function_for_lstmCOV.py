# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
from keras.models import Model
from keras.layers import Input, Activation, Dense, LSTM, Dropout, concatenate, GRU
from dbfread import DBF
#from keras.layers.recurrent import GRU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.utils.vis_utils import plot_model
import matplotlib.dates as mdate
import random
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
from keras.models import load_model

def mean_relative_error(y_true, y_pred):
    temp1 = np.abs(y_true[:, 0] - y_pred[:, 0]) / y_true[:, 0]
    #temp2 = np.abs(y_true[:, 1] - y_pred[:, 1]) / y_true[:, 1]
    temp1 = temp1[~np.isinf(temp1)]
    #temp2 = temp2[~np.isinf(temp2)]
    relative_error1 = np.nanmean(temp1, axis=0)
    #relative_error2 = np.nanmean(temp2, axis=0)
    #relative_error = np.nanmean([relative_error1, relative_error2])
    return relative_error1
    
def seed_tensorflow(seed):
    os.environ['PYTHONHASHSEED'] = str(seed) 
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1' # `pip install tensorflow
    
def data_process(dataset, dataDF):
    dataset2 = dataset.copy()
    dataDF2 = dataDF.copy()
    dataDF2.loc[dataDF2['county'] == 'New York City', 'state'] = 'New York1'
    dataset2.loc[dataset2['county'] == 'New York City', 'state'] = 'New York1'

    dataDF2.loc[dataDF2['county'] == 'Macomb', 'state'] = 'Michigan1'
    dataset2.loc[dataset2['county'] == 'Macomb', 'state'] = 'Michigan1'

    dataDF2.loc[dataDF2['county'] == 'Oakland', 'state'] = 'Michigan1'
    dataset2.loc[dataset2['county'] == 'Oakland', 'state'] = 'Michigan1'

    dataDF2.loc[(dataDF2['county'] == 'Wayne') & (dataDF2['state'] == 'Michigan'), 'state'] = 'Michigan1'
    dataset2.loc[(dataset2['county'] == 'Wayne') & (dataset2['state'] == 'Michigan'), 'state'] = 'Michigan1'

    dataDF2.loc[dataDF2['county'] == 'Cook', 'state'] = 'Illinois1'
    dataset2.loc[dataset2['county'] == 'Cook', 'state'] = 'Illinois1'

    dataDF2.loc[(dataDF2['county'] == 'Wayne') & (dataDF2['state'] == 'Illinois'), 'state'] = 'Illinois1'
    dataset2.loc[(dataset2['county'] == 'Wayne') & (dataset2['state'] == 'Illinois'), 'state'] = 'Illinois1'
    
    return dataset2, dataDF2

def data_process_inverse(dataset, dataDF):
    dataset2 = dataset.copy()
    dataDF2 = dataDF.copy()
    dataDF2.loc[dataDF2['state'] == 'New York1', 'state'] = 'New York'
    dataset2.loc[dataset2['state'] == 'New York1', 'state'] = 'New York'

    dataDF2.loc[dataDF2['state'] == 'Michigan1' , 'state'] = 'Michigan'
    dataset2.loc[dataset2['state'] == 'Michigan1', 'state'] = 'Michigan'

    dataDF2.loc[dataDF2['state'] == 'Illinois1', 'state'] = 'Illinois'
    dataset2.loc[dataset2['state'] == 'Illinois1', 'state'] = 'Illinois'

    return dataset2, dataDF2
    
def data_process_inverse2(dataset):
    
    dataset.loc[dataset['state'] == 'New York1', 'state'] = 'New York'

    dataset.loc[dataset['state'] == 'Michigan1', 'state'] = 'Michigan'

    dataset.loc[dataset['state'] == 'Illinois1', 'state'] = 'Illinois'

    return dataset 

def select_10_county(top10states, dataset, target = 1):
    _ = []
    datasetDF = dataset.sort_values(by = ['date'], axis = 0)
    tempTime = np.unique(datasetDF['date'])
    for s in top10states:
        dataset_temp = dataset[dataset['state'] == s]
        maxCountyList = dataset_temp[dataset_temp['date'] == tempTime[-1]]
        if target:
            maxCountyList = maxCountyList.sort_values(by = ['cum_dead'], axis = 0)
        else:
            maxCountyList = maxCountyList.sort_values(by = ['cum_confirm'], axis = 0)
        maxCounty = maxCountyList.StateCounty.tolist()[-1]
        _.append(maxCounty)
    return _



def get_scaler(dataset, dataDF, factorDF, latLongDF, look_back, top10states, target, remain_day = None):
    dataset = dataset.sort_values(by = ['date'], axis = 0)
    backupDF = dataset.reset_index(drop = True)
    dataset.reset_index(inplace = True, drop = True)
    tempTime = dataset['date'].unique()
    tempTime.sort()
    if remain_day:
        dataset = dataset[~dataset['date'].isin(tempTime[-remain_day:])]
        dataDF = dataDF[~dataDF['date'].isin(tempTime[-remain_day:])]
    
    dataset2, dataDF2 = data_process_inverse(dataset, dataDF)
    top_10_county = select_10_county(top10states, dataset2, target)
    
    #####################
    #create scaler for covid-19
    np.random.seed(7)
    
    dataDF.loc[dataDF['county'] == 'New York City', 'fips'] = '36061' 
    dataDF.dropna(axis = 0, subset = ['fips'], inplace = True)
    dataDF['fips'] = dataDF['fips'].astype(np.int64)
    
    dataDF['county'] = dataDF['county'].map(lambda x: x[:-4]+'City' if x[-4:] == 'city' else x)
    dataDF.loc[dataDF['county'] == 'Fairbanks North Star Borough', 'county'] = 'Fairbanks North Star'
    dataDF.loc[dataDF['county'] == 'Ketchikan Gateway Borough', 'county'] = 'Ketchikan Gateway'
    dataDF.loc[dataDF['county'] == 'Kenai Peninsula Borough', 'county'] = 'Kenai Peninsula'
    dataDF.loc[dataDF['county'] == 'Do?a Ana', 'county'] = 'Dona Ana'
    dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Juneau City and Borough'), 'county'] = 'Juneau'
    dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Matanuska-Susitna Borough'), 'county'] = 'Matanuska-Susitna'
    dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Bethel Census Area'), 'county'] = 'Bethel'
    dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Kodiak Island Borough'), 'county'] = 'Kodiak Island'
    dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Nome Census Area'), 'county'] = 'Nome'
    dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Petersburg Borough'), 'county'] = 'Petersburg'
    dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Prince of Wales-Hyder Census Area'), 'county'] = 'Prince of Wales-Hyder'
    dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Southeast Fairbanks Census Area'), 'county'] = 'Southeast Fairbanks'
    dataDF.loc[(dataDF['state'] == 'Alaska') & (dataDF['county'] == 'Yukon-Koyukuk Census Area'), 'county'] = 'Yukon-Koyukuk'
    
    StateCounty = dataDF2.apply(lambda x: x['state'] + ' ' + x['county'], axis = 1)
    dataDF.insert(0, 'StateCounty', StateCounty)
    
    latLongDF.drop_duplicates(subset = ['fips'], inplace = True)
    latLongDF = pd.merge(dataDF, latLongDF, left_on = ['fips'], right_on = ['fips'], how = 'left')
    print(latLongDF['StateCounty'])
    if target:
        latLongDF.drop(['county', 'state', 'fips', 'date', 'deaths', 'COUNTYNAME'], axis = 1, inplace = True)
    else:
        latLongDF.drop(['county', 'state', 'fips', 'date', 'cases', 'COUNTYNAME'], axis = 1, inplace = True)
    latLongDF.drop_duplicates(subset = ['StateCounty'], inplace = True)
    ##############
    
    dataDF = dataDF.sort_values(by = ['date'], axis = 0)
    state = np.unique(dataset['state'])
    scalerCOVID = []
    try:
        dataDF.drop(['Unnamed: 0'], axis = 1, inplace = True)
    except:
        pass
    for i in range(len(state)):
        scalerCOVID_ = MinMaxScaler(feature_range=(0, 1))
        dataDF_temp = dataDF[dataDF['state'] == state[i]]
        dataDF_temp.drop(['StateCounty', 'date', 'county', 'state', 'fips'], axis = 1, inplace = True)
        try:
            #print(i, state[i], "append")
            dataDF_temp = scalerCOVID_.fit_transform(dataDF_temp)
            scalerCOVID.append(scalerCOVID_)
        except:
            pass
    
    
    dataDF.drop(['StateCounty', 'date', 'county', 'state', 'fips'], axis = 1, inplace = True)
    dataDF.reset_index(inplace = True, drop = True)
    
    #########################
    #create scaler for factor
    np.random.seed(7)
    factorDF['Presence of Water Violation'] = factorDF['Presence of Water Violation'].map(lambda x: 1 if x == 'Yes' else (0 if x == 'No' else x))
    factorDF.fillna(factorDF.mean(), inplace = True)
    factorDF.drop(['County', 'State'], axis = 1, inplace = True)
    factorDF.reset_index(inplace = True, drop = True)
    factorColname_or = factorDF.columns.values.tolist()
    scalerFactor = MinMaxScaler(feature_range=(0, 1))
    factorDF = scalerFactor.fit_transform(factorDF)
    
    #########################
    #create scaler for lat-long
    np.random.seed(7)
    latLongScalerDF = latLongDF.drop(['StateCounty'], axis = 1)
    latLongScalerDF.reset_index(inplace = True, drop = True)
    
    scalerLatLong = MinMaxScaler(feature_range=(0, 1))
    try:
        latLongScalerDF.drop(['Unnamed: 0'], axis = 1, inplace = True)
    except:
        #print("'Unnamed: 0' not exists in latLongScalerDF")
        pass
    latLongScalerDF = scalerLatLong.fit_transform(latLongScalerDF)
    
    return top_10_county, scalerCOVID, scalerFactor, scalerLatLong, factorColname_or, latLongDF



def get_train_data(dataset, latLongDF, scalerCOVID, scalerFactor, scalerLatLong, look_back, target, remain_day = None, seed = 0):
    dataset = dataset.sort_values(by = ['date'], axis = 0)
    dataset.reset_index(inplace = True, drop = True)
    
    dataset_or = dataset.drop(['StateCounty', 'date', 'county', 'state'], axis = 1)
    
    dataset_or_dead = dataset_or.iloc[1, :look_back].values.tolist()
    if target:
        dataset_or_dead = dataset_or_dead + dataset_or.cum_dead.values.tolist()
    else:
        dataset_or_dead = dataset_or_dead + dataset_or.cum_confirm.values.tolist()
    
    dataset_or = np.array([dataset_or_dead])
    dataset_or = np.transpose(dataset_or, (1, 0))
    
    tempTime = np.unique(dataset['date'])
    tempTime.sort()
    
    
    dataset = pd.merge(dataset, latLongDF, 
                       left_on = ['StateCounty'], right_on = ['StateCounty'],
                       how = 'left')
    try:
        dataset.drop(['Unnamed: 0'], axis = 1, inplace = True)
    except:
        #print("line 229: 'Unnamed: 0' not exist", )
        pass
    if remain_day:
        valDF = dataset[dataset['date'].isin(tempTime[-remain_day:])]
        dataset = dataset[~dataset['date'].isin(tempTime[-remain_day:])]
        #print("temtime: ", tempTime[-remain_day:])
    else:
        valDF = dataset[dataset['date'].isin(tempTime[-remain_day:])]
    
    date = dataset['date']
    date = np.array(date)
    np.save('/public/home/tianting/lstm/data/date' + str(target) + '.npy', date)
    #print("===date written===")
    state = np.unique(dataset['state'])
    state_train = dataset['state']
    state_train = np.array(state_train)
    state_test = valDF['state']
    state_test = np.array(state_test)
    state_county = dataset['StateCounty']
    np.save('/public/home/tianting/lstm/data/state' + str(target) + '.npy', state_county)
    #print("===state written===")
    if target:
        cum = dataset['cum_dead']
    else:
        cum = dataset['cum_confirm']
    cum = np.array(cum)
    np.save('/public/home/tianting/lstm/data/cum' + str(target) + '.npy', cum)
    #print("===StateCounty written===")
    
    #StateCounty_remain = dataset.StateCounty
    #StateCounty_remain = StateCounty_remain.drop_duplicates()
    #print("valDF['date']", np.unique(valDF['date']))
    valDF.to_csv('/public/home/tianting/lstm/data/valDF' + str(seed) +'.csv', header = True, index = 0)
    valDF.drop(['StateCounty', 'date', 'county', 'state'], axis = 1, inplace = True)
    dataset.drop(['StateCounty', 'date', 'county', 'state'], axis = 1, inplace = True)
    
    dataset.fillna(dataset.mean(), inplace = True)
    valDF.fillna(valDF.mean(), inplace = True)
    
    colname = list(dataset.columns.values)
    try:
        colname.remove('LON')
        colname.remove('LAT')
    except:
        #print("colname: ", colname)
        pass
    trainMainDF = dataset[colname[:look_back]]
    #***********
    #print("dataset.columns: ", dataset.columns, dataset.shape)
    
    trainAuxDF = dataset[colname[look_back:-1] + ['LON', 'LAT']]
    #print("factor model, trainAuxDF: ", trainAuxDF.columns)
    trainYDF = dataset[colname[-1:]]
    #print("trainYDF: ", trainYDF.columns)
    valDF.reset_index(drop = True, inplace = True)
    valMainDF = valDF[colname[:look_back]]
    valAuxDF = valDF[colname[look_back:-1] + ['LON', 'LAT']]
    valYDF = valDF[colname[-1:]]
    #print("line 289 :", len(scalerCOVID))
    
    for j in range(len(state)):
        index_train = np.where(state_train == state[j])[0]
        index_test = np.where(state_test == state[j])[0]
        for i in range(look_back):
            trainMainDF.iloc[index_train, i:i+1] = scalerCOVID[j].transform(trainMainDF.iloc[index_train, i:i+1])
            valMainDF.iloc[index_test, i:i+1] = scalerCOVID[j].transform(valMainDF.iloc[index_test, i:i+1])
        trainYDF.iloc[index_train, :] = scalerCOVID[j].transform(trainYDF.iloc[index_train, :])
        valYDF.iloc[index_test, :] = scalerCOVID[j].transform(valYDF.iloc[index_test, :])

    trainAuxDF.iloc[:, :-2] = scalerFactor.transform(trainAuxDF.iloc[:, :-2])
    trainAuxDF.iloc[:, -2:] = scalerLatLong.transform(trainAuxDF.iloc[:, -2:])
    
    valAuxDF.iloc[:, :-2] = scalerFactor.transform(valAuxDF.iloc[:, :-2])
    valAuxDF.iloc[:, -2:] = scalerLatLong.transform(valAuxDF.iloc[:, -2:])
    
    return trainMainDF, trainAuxDF, trainYDF, valMainDF, valAuxDF, valYDF, dataset_or, state, state_train, state_test

  
def factor_model(trainMainDF, trainAuxDF, trainYDF, 
                 valMainDF, valAuxDF, valYDF,
                 dataset_or, path, scalerCOVID, scalerFactor, 
                 look_back, remain_day, state, state_train, state_test, stayCol, model_type = "model", seed = 0, n_epoch = 200):
    trainAuxDF1 = trainAuxDF[stayCol]
    valAuxDF1 = valAuxDF[stayCol]

    trainMainDF = trainMainDF.fillna(0)
    trainAuxDF1 = trainAuxDF1.fillna(0)
    trainYDF = trainYDF.fillna(0)
    valMainDF = valMainDF.fillna(0)
    valAuxDF1 = valAuxDF1.fillna(0)
    valYDF = valYDF.fillna(0)
    train_main_x = trainMainDF.values
    train_aux_x = trainAuxDF1.values
    train_y = trainYDF.values
    
    
    val_main_x = valMainDF.values
    val_aux_x = valAuxDF1.values
    val_y = valYDF.values
    
    train_main_x = train_main_x.reshape(train_main_x.shape[0], 1, train_main_x.shape[1])
    val_main_x = val_main_x.reshape(val_main_x.shape[0], 1, val_main_x.shape[1])
    seed_tensorflow(seed)
    channels = seed

    
    ############################
    ######## network
    ############################
    batch_size = 32
    
    mainInputs = Input(shape=[train_main_x.shape[1], train_main_x.shape[2]], name = 'main_input')
    if model_type == "model": 
        x1 = GRU(50, input_shape = (train_main_x.shape[1], train_main_x.shape[2]), name = 'gru_1')(mainInputs)
        x11 = GRU(50, input_shape = (train_main_x.shape[1], train_main_x.shape[2]), name = 'gru_2')(mainInputs)
        x111 = GRU(50, input_shape = (train_main_x.shape[1], train_main_x.shape[2]), name = 'gru_3')(mainInputs)
        x1 = Dropout(0.3, name = 'dropout_1')(x1)
        x11 = Dropout(0.3, name = 'dropout_11')(x11)
        x111 = Dropout(0.3, name = 'dropout_111')(x111)
        x2 = LSTM(50, input_shape = [train_main_x.shape[1], train_main_x.shape[2]], name = 'lstm_1')(mainInputs)
        x22 = LSTM(50, input_shape = [train_main_x.shape[1], train_main_x.shape[2]], name = 'lstm_2')(mainInputs)
        x222 = LSTM(50, input_shape = [train_main_x.shape[1], train_main_x.shape[2]], name = 'lstm_3')(mainInputs)
        x2 = Dropout(0.3, name = 'dropout_2')(x2)
        x22 = Dropout(0.3, name = 'dropout_22')(x22)
        x222 = Dropout(0.3, name = 'dropout_222')(x222)
        x3 = concatenate([x1, x11, x111, x2, x22, x222], axis = 1, name = 'concatenate_1')
        #x3 = concatenate([x1, x2], axis = 1, name = 'concatenate_1')
        
        auxInputs = Input(shape = (train_aux_x.shape[1],), name = 'aux_input')
        x4 = concatenate([x3, auxInputs], name = 'concatenate_2')
        
        x5 = Dense(train_y.shape[1], name = 'dense_1')(x4)
        x5 = Activation("relu", name = 'activation_1')(x5)
        
        model = Model([mainInputs, auxInputs], x5)
        
    elif model_type == "GRU":

        x1 = GRU(50, input_shape = (train_main_x.shape[1], train_main_x.shape[2]), name = 'gru_1')(mainInputs)
        x11 = GRU(50, input_shape = (train_main_x.shape[1], train_main_x.shape[2]), name = 'gru_2')(mainInputs)
        x111 = GRU(50, input_shape = (train_main_x.shape[1], train_main_x.shape[2]), name = 'gru_3')(mainInputs)
        x1 = Dropout(0.3, name = 'dropout_1')(x1)
        x11 = Dropout(0.3, name = 'dropout_11')(x11)
        x111 = Dropout(0.3, name = 'dropout_111')(x111)
        auxInputs = Input(shape = (train_aux_x.shape[1],), name = 'aux_input')
        x2 = concatenate([x1, x11, x111, auxInputs], name = 'concatenate_1')
        x3 = Dense(train_y.shape[1], name = 'dense_1')(x2)
        x3 = Activation("relu", name = 'activation_1')(x3)
        model = Model([mainInputs, auxInputs], x3) 
        
    elif model_type == "LSTM":
        
        x1 = LSTM(50, input_shape = [train_main_x.shape[1], train_main_x.shape[2]], name = 'lstm_1')(mainInputs)
        x11 = LSTM(50, input_shape = [train_main_x.shape[1], train_main_x.shape[2]], name = 'lstm_2')(mainInputs)
        x111 = LSTM(50, input_shape = [train_main_x.shape[1], train_main_x.shape[2]], name = 'lstm_3')(mainInputs)
        
        x1 = Dropout(0.3, name = 'dropout_1')(x1)
        x11 = Dropout(0.3, name = 'dropout_11')(x11)
        x111 = Dropout(0.3, name = 'dropout_111')(x111)
        
        auxInputs = Input(shape = (train_aux_x.shape[1],), name = 'aux_input')
        x2 = concatenate([x1, x11, x111, auxInputs], name = 'concatenate_1')
        x3 = Dense(train_y.shape[1], name = 'dense_1')(x2)
        x3 = Activation("relu", name = 'activation_1')(x3)
        model = Model([mainInputs, auxInputs], x3)
    
    model.compile(loss= 'mean_squared_error', optimizer="adam", metrics=["accuracy"])
    reduce_lr = ReduceLROnPlateau(factor=0.3, patience=10, min_lr=0.000001, verbose=1)
    early_stopping = EarlyStopping(patience=50, verbose=1)

    print("history")
    print("model fit: ", train_main_x.shape, train_aux_x.shape)
    history = model.fit({'main_input': train_main_x, 'aux_input': train_aux_x}, 
                    train_y, epochs = n_epoch, batch_size = batch_size,
                    validation_data = ([val_main_x, val_aux_x], val_y), 
                    verbose = 2, shuffle = False,
                    callbacks = [reduce_lr, early_stopping])
    
    
    print("model save")
    #model.save(path + '/weights/lstm_county_' + str(seed) +  '_' + str(len(stayCol)) + '.h5') 
    
    model = load_model(path + 'weights/lstm_county_23_5.h5')
    #########################
    ############ predict
    #########################
    
    predict = []
    
    trainPredict = model.predict([train_main_x, train_aux_x], batch_size = batch_size)
    
    for c in range(len(state)):
        index = np.where(state_train == state[c])[0]
        trainPredict[index] = scalerCOVID[c].inverse_transform(trainPredict[index])
        
    trainPredict = trainPredict.astype(np.int32)
    
    trainScore = math.sqrt(mean_squared_error(train_y, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    
    valPredict = model.predict([val_main_x, val_aux_x], batch_size = batch_size)
    print("valPredict: ", valPredict.shape)
    for c in range(len(state)):
        index = np.where(state_test == state[c])[0]
        try:
            valPredict[index] = scalerCOVID[c].inverse_transform(valPredict[index])
            val_y[index] = scalerCOVID[c].inverse_transform(val_y[index])
        except:
            print("valPredict Error: ", state[c])
    valPredict = valPredict.astype(np.int32)
    np.save(('/public/home/tianting/lstm/data/valPredict_' + model_type +  '_') + str(seed) + '.npy', valPredict)
    np.save('/public/home/tianting/lstm/data/val_Y_' + model_type + '.npy', val_y)
    print("=== val written ===")
    return model
    
    
def get_county_data(datasetDF, latLongDF, stayCol, state,
                    scalerCOVID, scalerFactor, scalerLatLong, 
                    stateCountyName, look_back = 14, remain_day = 60, sd = None):
                    
    dataset = datasetDF.sort_values(by = ['date'], axis = 0)
    if sd:
        dataset = dataset[dataset['date'] <= sd]
    else:
        print("sd none")
    dataset.reset_index(inplace = True, drop = True)
    dataset = dataset[dataset['StateCounty'] == stateCountyName]
    
    dataset = dataset.reset_index()
    
    stateName = dataset['state'][0]
    
    tempTime = dataset['date'].unique()
    tempTime.sort()
    
    #dataset = dataset.drop(['StateCounty', 'date', 'county', 'state'], axis = 1)
    dataset = pd.merge(dataset, latLongDF, left_on = ['StateCounty'], right_on = ['StateCounty'], how = 'left')
    
    if remain_day:
        dataset = dataset[dataset['date'].isin(tempTime[-remain_day:])]
    date = dataset['date']
    try:
        dataset.drop(['Unnamed: 0'], axis = 1, inplace = True)
    except:
        #print("'Unnamed: 0' not in dataset")
        pass
    dataset.drop(['StateCounty', 'date', 'county', 'state'], axis = 1, inplace = True)
    dataset.fillna(0, inplace = True)
    
    colname = list(dataset.columns.values)
    colname.remove('index')
    colname.remove('LON')
    colname.remove('LAT')
    trainMainDF = dataset[colname[:look_back]]
    trainAuxDF = dataset[colname[look_back:-1] + ['LON', 'LAT']]

    trainYDF = dataset[colname[-1:]]
    print(trainYDF.columns)
    index = np.where(state == stateName)[0]
    scaler_ = scalerCOVID[index[0]]
    trainYDF = scaler_.transform(trainYDF)
    for i in range(look_back):
        trainMainDF.iloc[:, i:(i+1)] = scaler_.transform(trainMainDF.iloc[:, i:(i+1)])
    trainAuxDF.iloc[:, :-2] = scalerFactor.transform(trainAuxDF.iloc[:, :-2])
    trainAuxDF.iloc[:, -2:] = scalerLatLong.transform(trainAuxDF.iloc[:, -2:])
    
    trainAuxDF = trainAuxDF[stayCol]
    return trainMainDF, trainAuxDF, trainYDF, date, stateName
    

def predict_county(model, save_path, trainMainDF, trainAuxDF, trainYDF, remain_day, scalerCOVID, state, stateName, look_back, start_index = 0, change_factor_index = 0, change_factor = 1):   
    if start_index + 50 > len(trainYDF):
        print("no true value error...")
        return np.zeros(30), np.zeros(30)
    else:
    
        start_index = int(start_index)
        testx = [0.]*(51)
        #testx[0:14] = trainMainDF.iloc[start_index, :]
        for i in range(look_back):
            testx[i] = trainMainDF.iloc[start_index, i]
        for i in range(7):
            testx[look_back + i] = trainMainDF.iloc[i + 1 + start_index, -1]

        testx_factor = trainAuxDF.iloc[0, :].copy()
        testx_factor[change_factor_index] = testx_factor[change_factor_index] * change_factor
        testx_factor = np.array(testx_factor)
        testx_factor = testx_factor.reshape(1, -1)
        for i in range(30):
            testxx = testx[i:(i+look_back)]
            testxx = np.reshape(testxx, (1, 1, look_back))
            testy = model.predict([testxx, testx_factor], verbose = 0)
            #print(i, testy)
            testx[look_back + 7 + i] = testy.tolist()[0][0]
    
        testx = np.array(testx).reshape(-1, 1)
        index = np.where(state == stateName)[0]
        scaler_ = scalerCOVID[index[0]]
        testx = scaler_.inverse_transform(testx)
        trainYDF = scaler_.inverse_transform(trainYDF)
        print(testx)
        return testx, trainYDF
    
        return testx, trainYDF
