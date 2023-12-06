# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt 

path = '.~\\data\\'
valDF = pd.read_csv(path + "valDF2023.csv", parse_dates = ['date'] ,infer_datetime_format=True)
valDF = valDF.reset_index(drop = True)
valDF['date'] = [str(valDF['date'][i]).split(' ')[0] for i in range(len(valDF))]
date = np.unique(valDF['date'])
date_ = []

top10 = True

for d in date:
    year, month, day = time.strptime(d, "%Y-%m-%d")[:3]
    day = datetime.date(year, month, day)
    if day.weekday() == 5:
        date_.append(d)
    
date_ = date
loc = pd.read_csv(path + "locations.csv")
loc = loc[loc['location'] != 'US']
loc['location'] = loc['location'].astype(int)
loc = loc[loc['location'] <= 100]
loc = loc.drop('population', axis=1)
loc = loc.rename(columns={'location_name':'state'})



f_select = pd.merge(loc, valDF, on = 'state')
f_select = f_select[f_select['date'] == date_[-1]]
f_select = f_select.dropna(how='any', axis=0)
f_select = f_select.sort_values(by = 'cum_dead')
f_select = f_select.reset_index(drop = True)
fips_ = f_select['location'][-10:]
mae_ = []
mre_ = []
pre_method = ['model']
seed_list = range(50)
for i in seed_list:
    try:
        valpre = np.load(path +"valPredict_" + pre_method[0] + "_" + str(i) + ".npy")
        valDF = pd.read_csv(path + "valDF" + str(i) + ".csv", parse_dates = ['date'] ,infer_datetime_format=True)
        valDF['date'] = [str(valDF['date'][i]).split(' ')[0] for i in range(len(valDF))]
        valDF = valDF[['state', 'cum_confirm', 'date']]
        valDF['pre'] = valpre[0:len(valDF), 0]
        print(i)
        print(valpre.shape, len(valDF))
        valDF = valDF[np.in1d(valDF['date'], date_)]
        df = pd.DataFrame([])

        state = np.unique(valDF['state'])
        dead_pre = []
        dead = []
        date_list = []
        state_list = []
        for s in state: 
            for d in date_:
                data = valDF[valDF['state'] == s]
                data = data[data['date'] == d]
                dead.append(sum(data['cum_confirm']))
                dead_pre.append(sum(data['pre']))
                date_list.append(d)
                state_list.append(s)


        dead = np.array(dead)
        dead_pre = np.array(dead_pre)

        df = pd.DataFrame([])
        df['true'] = dead
        df['pre'] = dead_pre
        df['state'] = state_list
        df['date'] = date_list
        df = pd.merge(df, loc, on = 'state')
        df.loc[df['pre'] < 0, 'pre'] = 0
        if top10:
            df = df[np.in1d(df['location'], fips_)]
        df['pre'] = df['pre']
        #df = df[df['state'] != 'New York']
        df['ab error'] =  np.abs(df['true'] - df['pre'])
        mae = np.mean(np.abs(df['true']-df['pre']))
        mre = np.mean(np.abs(df['true']-df['pre'])/df['true'])
        
        mae_.append(mae)
        mre_.append(mre)
    except:
        pass

mae = np.array(mae_)
mre = np.array(mre_)
mre = mre[~np.isnan(mae)]
mae = mae[~np.isnan(mae)]

mre = mre[mae < 1000]
mae = mae[mae < 1000]
print(np.mean(mae), np.sqrt(np.var(mae)))
print(np.mean(mre), np.sqrt(np.var(mre)))

sc = np.unique(df['state'])
for c in sc:
    df_temp = df[df['state'] == c]
    print(c)
    print(np.mean(np.abs(df_temp['true'] - df_temp['pre'])/df_temp['true']))
    
    
    
    
    
    
    
