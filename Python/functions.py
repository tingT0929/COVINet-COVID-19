# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:24:39 2020

@author: Yuting Zhang
"""

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Activation, Dense, LSTM, Dropout, concatenate
from keras.layers.recurrent import GRU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
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

def get_scaler(dataset, dataDF, factorDF, latLongDF, look_back, remain_day = None):
    dataset = dataset.sort_values(by = ['date'], axis = 0)
    backupDF = dataset.reset_index(drop = True)
    dataset.reset_index(inplace = True, drop = True)
    
    tempTime = dataset['date'].unique()
    tempTime.sort()
    if remain_day:
        dataset = dataset[~dataset['date'].isin(tempTime[-remain_day:])]
        dataDF = dataDF[~dataDF['date'].isin(tempTime[-remain_day:])]
    
    StateCounty_remain = dataset.StateCounty
    StateCounty_remain = StateCounty_remain.drop_duplicates()
    
    dataset.drop(['StateCounty', 'date', 'county', 'state'], axis = 1, inplace = True)
    #####################
    #create scaler for covid-19
    np.random.seed(7)
     
    dataDF.loc[dataDF['county'] == 'New York City', 'fips'] = '36061' 
    dataDF.dropna(axis = 0, subset = ['fips'], inplace = True)
    dataDF['fips'] = dataDF['fips'].astype(np.int64)
    
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
    
    latLongDF.drop_duplicates(subset = ['fips'], inplace = True)
    latLongDF = pd.merge(dataDF, latLongDF, left_on = ['fips'], right_on = ['fips'], how = 'left')
    latLongDF.drop(['county', 'state', 'fips', 'date', 'cases', 'deaths', 'COUNTYNAME'], axis = 1, inplace = True)
    latLongDF.drop_duplicates(subset = ['StateCounty'], inplace = True)
    ##############
    
    dataDF = dataDF.sort_values(by = ['date'], axis = 0)
    dataDF = dataDF[dataDF.StateCounty.isin(backupDF.StateCounty)]
    dataDF.drop(['StateCounty', 'date', 'county', 'state', 'fips'], axis = 1, inplace = True)
    dataDF.reset_index(inplace = True, drop = True)
    
    # Normalization
    scalerCOVID = MinMaxScaler(feature_range=(0, 1))
    dataDF = scalerCOVID.fit_transform(dataDF)
    
    #########################
    #create scaler for factor
    np.random.seed(7)
    factorDF['Presence of Water Violation'] = factorDF['Presence of Water Violation'].map(lambda x: 1 if x == 'Yes' else (0 if x == 'No' else x))
    factorDF.fillna(factorDF.mean(), inplace = True)
    factorDF.drop(['County', 'State'], axis = 1, inplace = True)
    factorDF.reset_index(inplace = True, drop = True)
    factorColname_or = factorDF.columns.values.tolist()
    
    # Normalization
    scalerFactor = MinMaxScaler(feature_range=(0, 1))
    factorDF = scalerFactor.fit_transform(factorDF)
    
    #########################
    #create scaler for lat-long
    np.random.seed(7)
    latLongScalerDF = latLongDF.drop(['StateCounty'], axis = 1)
    latLongScalerDF.reset_index(inplace = True, drop = True)
    
    # Normalization
    scalerLatLong = MinMaxScaler(feature_range=(0, 1))
    latLongScalerDF = scalerLatLong.fit_transform(latLongScalerDF)
    
    return scalerCOVID, scalerFactor, scalerLatLong, factorColname_or, latLongDF

def get_train_data(dataset, latLongDF, scalerCOVID, scalerFactor, scalerLatLong, look_back, remain_day = None):
    dataset = dataset.sort_values(by = ['date'], axis = 0)
    dataset.reset_index(inplace = True, drop = True)
   
    dataset_or = dataset.drop(['StateCounty', 'date', 'county', 'state'], axis = 1)
    dataset_or_con = dataset_or.iloc[1, :look_back*2:2].values.tolist()
    dataset_or_con = dataset_or_con + dataset_or.cum_confirm.values.tolist()
    
    dataset_or_dead = dataset_or.iloc[1, 1:look_back*2:2].values.tolist()
    dataset_or_dead = dataset_or_dead + dataset_or.cum_dead.values.tolist()
    
    dataset_or = np.array([dataset_or_con, dataset_or_dead])
    dataset_or = np.transpose(dataset_or, (1, 0))
    
    tempTime = dataset['date'].unique()
    tempTime.sort()
    dataset = pd.merge(dataset, latLongDF, 
                       left_on = ['StateCounty'], right_on = ['StateCounty'],
                       how = 'left')
    if remain_day:
        valDF = dataset[dataset['date'].isin(tempTime[-remain_day:])]
        dataset = dataset[~dataset['date'].isin(tempTime[-remain_day:])]
    else:
        valDF = dataset[dataset['date'].isin(tempTime[-look_back:])]
    
    StateCounty_remain = dataset.StateCounty
    StateCounty_remain = StateCounty_remain.drop_duplicates()
    valDF.drop(['StateCounty', 'date', 'county', 'state'], axis = 1, inplace = True)
    dataset.drop(['StateCounty', 'date', 'county', 'state'], axis = 1, inplace = True)
    
    dataset.fillna(dataset.mean(), inplace = True)
    valDF.fillna(valDF.mean(), inplace = True)
    
    colname = list(dataset.columns.values)
    colname.remove('LON')
    colname.remove('LAT')
    trainMainDF = dataset[colname[:2*look_back]]
    trainAuxDF = dataset[colname[2*look_back:-2] + ['LON', 'LAT']]
    trainYDF = dataset[colname[-2:]]
    valDF.reset_index(drop = True, inplace = True)
    valMainDF = valDF[colname[:2*look_back]]
    valAuxDF = valDF[colname[2*look_back:-2] + ['LON', 'LAT']]
    valYDF = valDF[colname[-2:]]
    
    for i in range(look_back):
        trainMainDF.iloc[:, 2*i:2*i+2] = scalerCOVID.transform(trainMainDF.iloc[:, 2*i:2*i+2])
        valMainDF.iloc[:, 2*i:2*i+2] = scalerCOVID.transform(valMainDF.iloc[:, 2*i:2*i+2])
    trainAuxDF.iloc[:, :-2] = scalerFactor.transform(trainAuxDF.iloc[:, :-2])
    trainAuxDF.iloc[:, -2:] = scalerLatLong.transform(trainAuxDF.iloc[:, -2:])
    trainYDF.iloc[:, :] = scalerCOVID.transform(trainYDF.iloc[:, :])
    valAuxDF.iloc[:, :-2] = scalerFactor.transform(valAuxDF.iloc[:, :-2])
    valAuxDF.iloc[:, -2:] = scalerLatLong.transform(valAuxDF.iloc[:, -2:])
    valYDF.iloc[:, :] = scalerCOVID.transform(valYDF.iloc[:, :])
    return trainMainDF, trainAuxDF, trainYDF, valMainDF, valAuxDF, valYDF, dataset_or

def training_model(trainMainDF, trainAuxDF, trainYDF, 
                   valMainDF, valAuxDF, valYDF, 
                   factorColname, save_path, weight_file,
                   ny_weight_file, noNewYork = 1, isTrain = 1):
    trainAuxDF1 = trainAuxDF.drop(factorColname, axis = 1) 
    valAuxDF1 = valAuxDF.drop(factorColname, axis = 1) 
    
    train_main_x = trainMainDF.values
    train_aux_x = trainAuxDF1.values
    train_y = trainYDF.values
    
    val_main_x = valMainDF.values
    val_aux_x = valAuxDF1.values
    val_y = valYDF.values
    
    train_main_x = train_main_x.reshape(train_main_x.shape[0], 1, train_main_x.shape[1])
    val_main_x = val_main_x.reshape(val_main_x.shape[0], 1, val_main_x.shape[1])
    ############################
    ######## network
    ############################
    batch_size = 32
    epochs = 100
    
    alpha1 = 0.05
    alpha2 = 0.2
    
    mainInputs = Input(shape=[train_main_x.shape[1], train_main_x.shape[2]], name = 'main_input')
    x1 = GRU(50, input_shape = (train_main_x.shape[1], train_main_x.shape[2]), name = 'gru_1')(mainInputs)
    x1 = Dropout(alpha1, name = 'dropout_1')(x1)
    x2 = LSTM(50, input_shape = [train_main_x.shape[1], train_main_x.shape[2]], name = 'lstm_1')(mainInputs)
    x2 = Dropout(alpha2, name = 'dropout_2')(x2)
    x3 = concatenate([x1, x2], axis = 1, name = 'concatenate_1')
    
    auxInputs = Input(shape = (train_aux_x.shape[1],), name = 'aux_input')
    x4 = concatenate([x3, auxInputs], name = 'concatenate_2')
    
    x5 = Dense(train_y.shape[1], name = 'dense_1')(x4)
    x5 = Activation("relu", name = 'activation_1')(x5)
    model = Model([mainInputs, auxInputs], x5)
    
    model.compile(loss='mean_squared_error', optimizer="adam", metrics=["accuracy"])
    plot_model(model, to_file = save_path + '/model.png')#, show_shapes=True)
    
    if noNewYork == 1:
        model.load_weights(ny_weight_file)
        epochs = 100

    early_stopping = EarlyStopping(patience=40, verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=0.3, patience=10, min_lr=0.00001, verbose=1)
    checkpointer = ModelCheckpoint(filepath = weight_file, verbose=1, 
                                   monitor='val_loss', mode='auto', save_best_only=True)
    
    if isTrain == 1:
        history = model.fit({'main_input': train_main_x, 'aux_input': train_aux_x}, 
                        train_y, epochs = epochs, batch_size = batch_size,
                        validation_data = ([val_main_x, val_aux_x], val_y), 
                        verbose = 1, shuffle = False,
                        callbacks = [reduce_lr, checkpointer, early_stopping])
        
        plt.plot(history.history['loss'], label = 'train')
        plt.plot(history.history['val_loss'], label = 'val')
        plt.legend()
        plt.show()
    else:
        pass
    return model, train_main_x, train_aux_x, train_y, val_main_x, val_aux_x, val_y

def training_GRU_model(trainMainDF, trainAuxDF, trainYDF, 
                   valMainDF, valAuxDF, valYDF, 
                   factorColname, save_path, weight_file, isTrain = 1):
    trainAuxDF1 = trainAuxDF.drop(factorColname, axis = 1) 
    valAuxDF1 = valAuxDF.drop(factorColname, axis = 1) 
    
    train_main_x = trainMainDF.values
    train_aux_x = trainAuxDF1.values
    train_y = trainYDF.values
    
    val_main_x = valMainDF.values
    val_aux_x = valAuxDF1.values
    val_y = valYDF.values
    
    train_main_x = train_main_x.reshape(train_main_x.shape[0], 1, train_main_x.shape[1])
    val_main_x = val_main_x.reshape(val_main_x.shape[0], 1, val_main_x.shape[1])
    ############################
    ######## network
    ############################
    batch_size = 32
    epochs = 100
    
    alpha1 = 0.05
    
    mainInputs = Input(shape=[train_main_x.shape[1], train_main_x.shape[2]], name = 'main_input')
    x1 = GRU(50, input_shape = (train_main_x.shape[1], train_main_x.shape[2]), name = 'gru_1')(mainInputs)
    x1 = Dropout(alpha1, name = 'dropout_1')(x1)
    
    auxInputs = Input(shape = (train_aux_x.shape[1],), name = 'aux_input')
    x2 = concatenate([x1, auxInputs], name = 'concatenate_1')
    
    x3 = Dense(train_y.shape[1], name = 'dense_1')(x2)
    x3 = Activation("relu", name = 'activation_1')(x3)
    model = Model([mainInputs, auxInputs], x3)
    
    model.compile(loss='mean_squared_error', optimizer="adam", metrics=["accuracy"])
    plot_model(model, to_file = save_path + '/model.png')#, show_shapes=True)
    
    early_stopping = EarlyStopping(patience=40, verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=0.3, patience=10, min_lr=0.00001, verbose=1)
    checkpointer = ModelCheckpoint(filepath = weight_file, verbose=1, 
                                   monitor='val_loss', mode='auto', save_best_only=True)
    
    if isTrain == 1:
        history = model.fit({'main_input': train_main_x, 'aux_input': train_aux_x}, 
                        train_y, epochs = epochs, batch_size = batch_size,
                        validation_data = ([val_main_x, val_aux_x], val_y), 
                        verbose = 1, shuffle = False,
                        callbacks = [reduce_lr, checkpointer, early_stopping])
        
        plt.plot(history.history['loss'], label = 'train')
        plt.plot(history.history['val_loss'], label = 'val')
        plt.legend()
        plt.show()
    else:
        pass
    return model, train_main_x, train_aux_x, train_y, val_main_x, val_aux_x, val_y

def training_lstm_model(trainMainDF, trainAuxDF, trainYDF, 
                   valMainDF, valAuxDF, valYDF, 
                   factorColname, save_path, weight_file, isTrain = 1):
    trainAuxDF1 = trainAuxDF.drop(factorColname, axis = 1) 
    valAuxDF1 = valAuxDF.drop(factorColname, axis = 1) 
    
    train_main_x = trainMainDF.values
    train_aux_x = trainAuxDF1.values
    train_y = trainYDF.values
    
    val_main_x = valMainDF.values
    val_aux_x = valAuxDF1.values
    val_y = valYDF.values
    
    train_main_x = train_main_x.reshape(train_main_x.shape[0], 1, train_main_x.shape[1])
    val_main_x = val_main_x.reshape(val_main_x.shape[0], 1, val_main_x.shape[1])
    ############################
    ######## network
    ############################
    batch_size = 32
    epochs = 100
    
    alpha2 = 0.2

    mainInputs = Input(shape=[train_main_x.shape[1], train_main_x.shape[2]], name = 'main_input')
    x1 = LSTM(50, input_shape = [train_main_x.shape[1], train_main_x.shape[2]], name = 'lstm_1')(mainInputs)
    x1 = Dropout(alpha2, name = 'dropout_1')(x1)
    
    auxInputs = Input(shape = (train_aux_x.shape[1],), name = 'aux_input')
    x2 = concatenate([x1, auxInputs], name = 'concatenate_1')
    
    x3 = Dense(train_y.shape[1], name = 'dense_1')(x2)
    x3 = Activation("relu", name = 'activation_1')(x3)
    model = Model([mainInputs, auxInputs], x3)
    
    model.compile(loss='mean_squared_error', optimizer="adam", metrics=["accuracy"])
    plot_model(model, to_file = save_path + '/model.png')#, show_shapes=True)
    
    early_stopping = EarlyStopping(patience=40, verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=0.3, patience=10, min_lr=0.00001, verbose=1)
    checkpointer = ModelCheckpoint(filepath = weight_file, verbose=1, 
                                   monitor='val_loss', mode='auto', save_best_only=True)
    
    if isTrain == 1:
        history = model.fit({'main_input': train_main_x, 'aux_input': train_aux_x}, 
                        train_y, epochs = epochs, batch_size = batch_size,
                        validation_data = ([val_main_x, val_aux_x], val_y), 
                        verbose = 1, shuffle = False,
                        callbacks = [reduce_lr, checkpointer, early_stopping])
        
        plt.plot(history.history['loss'], label = 'train')
        plt.plot(history.history['val_loss'], label = 'val')
        plt.legend()
        plt.show()
    else:
        pass
    return model, train_main_x, train_aux_x, train_y, val_main_x, val_aux_x, val_y
    