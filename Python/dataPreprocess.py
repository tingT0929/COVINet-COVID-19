# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 21:08:21 2020

@author: Yuting Zhang
"""

import os
import numpy as np
import pandas as pd
from datetime import timedelta

path = './data'
os.chdir(path)

###################################################
################### load data #####################
###################################################
dataDF = pd.read_csv('us-counties.csv')
StateCounty = dataDF.apply(lambda x: x['state'] + ' ' + x['county'], axis = 1)
dataDF.insert(0, 'StateCounty', StateCounty)
dataDF['date'] = pd.to_datetime(dataDF['date'], format = '%Y-%m-%d')
dataDF = dataDF.sort_values(by = ['date'], axis = 0)
countyList = StateCounty.unique().tolist()

###################################################
################## check data #####################
###################################################
for j in range(len(countyList)):
    c = countyList[j]
    countyTime = dataDF[dataDF['StateCounty'] == c]['date'].tolist()
    for i in range(1, len(countyTime)):
        if i < len(countyTime) - 1:
            lastCase = dataDF.loc[(dataDF['date'] == countyTime[i - 1]) & (dataDF['StateCounty'] == c)]['cases'].values[0]
            nowCase = dataDF.loc[(dataDF['date'] == countyTime[i]) & (dataDF['StateCounty'] == c)]['cases'].values[0]
            nextCase = dataDF.loc[(dataDF['date'] == countyTime[i + 1]) & (dataDF['StateCounty'] == c)]['cases'].values[0]
            
            lastDead = dataDF.loc[(dataDF['date'] == countyTime[i - 1]) & (dataDF['StateCounty'] == c)]['deaths'].values[0]
            nowDead = dataDF.loc[(dataDF['date'] == countyTime[i]) & (dataDF['StateCounty'] == c)]['deaths'].values[0]
            nextDead = dataDF.loc[(dataDF['date'] == countyTime[i + 1]) & (dataDF['StateCounty'] == c)]['deaths'].values[0]            

            if nowCase < lastCase:
                nowCase = max(int(0.5 * (lastCase + nextCase)), lastCase)
                dataDF.loc[(dataDF['date'] == countyTime[i]) & (dataDF['StateCounty'] == c), 'cases'] = nowCase
            if nowDead < lastDead:
                nowDead = max(int(0.5 * (lastDead + nextDead)), lastDead)
                dataDF.loc[(dataDF['date'] == countyTime[i]) & (dataDF['StateCounty'] == c), 'deaths'] = nowDead

        elif i == len(countyTime) - 1:
            lastCase = dataDF.loc[(dataDF['date'] == countyTime[i - 1]) & (dataDF['StateCounty'] == c)]['cases'].values[0]
            nowCase = dataDF.loc[(dataDF['date'] == countyTime[i]) & (dataDF['StateCounty'] == c)]['cases'].values[0]
            
            lastDead = dataDF.loc[(dataDF['date'] == countyTime[i - 1]) & (dataDF['StateCounty'] == c)]['deaths'].values[0]
            nowDead = dataDF.loc[(dataDF['date'] == countyTime[i]) & (dataDF['StateCounty'] == c)]['deaths'].values[0]
            if nowCase < lastCase:
                nowCase = lastCase
                dataDF.loc[(dataDF['date'] == countyTime[i]) & (dataDF['StateCounty'] == c), 'cases'] = nowCase
            if nowDead < lastDead:
                nowDead = lastDead
                dataDF.loc[(dataDF['date'] == countyTime[i]) & (dataDF['StateCounty'] == c), 'deaths'] = nowDead

dataDF.drop(['StateCounty'], axis = 1, inplace = True)
dataDF = dataDF.sort_values(by = ['date','county'], axis = 0)
dataDF.reset_index(drop = True, inplace = True)
dataDF.to_csv('us-counties-processed.csv', encoding = 'utf_8_sig', index = False)

###################################################
################# data smoothing ##################
###################################################
def moving_average(interval, window_size, method = 'same'):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, method)

for j in range(len(countyList)):
    c = countyList[j]
    tempCase = dataDF[dataDF['StateCounty'] == c]['cases'].tolist()
    tempCase = np.asarray(tempCase)
    tempDeath = dataDF[dataDF['StateCounty'] == c]['deaths'].tolist()
    tempDeath = np.asarray(tempDeath)
    tempCase1 = moving_average(interval = tempCase, window_size = 7, method = 'valid')
    tempDeath1 = moving_average(interval = tempDeath, window_size = 7, method = 'valid')

    if (len(tempCase1) <= 2) or (len(tempCase) <= 6):
        continue
    tempCase[3:-3] = tempCase1
    tempDeath[3:-3] = tempDeath1
    tempCase2 = moving_average(interval = tempCase, window_size = 2, method = 'same')
    tempDeath2 = moving_average(interval = tempDeath, window_size = 2, method = 'same')

    dataDF.loc[dataDF['StateCounty'] == c, 'cases'] = tempCase2.tolist()
    dataDF.loc[dataDF['StateCounty'] == c, 'deaths'] = tempDeath2.tolist()
    dataDF.loc[dataDF['StateCounty'] == c, ['cases', 'deaths']] = dataDF.loc[dataDF['StateCounty'] == c, ['cases', 'deaths']].astype(np.int32)

    countyTime = dataDF[dataDF['StateCounty'] == c]['date'].tolist()
    for i in range(1, len(countyTime)):
        if i < len(countyTime) - 1:
            lastCase = dataDF.loc[(dataDF['date'] == countyTime[i - 1]) & (dataDF['StateCounty'] == c)]['cases'].values[0]
            nowCase = dataDF.loc[(dataDF['date'] == countyTime[i]) & (dataDF['StateCounty'] == c)]['cases'].values[0]
            nextCase = dataDF.loc[(dataDF['date'] == countyTime[i + 1]) & (dataDF['StateCounty'] == c)]['cases'].values[0]
            
            lastDead = dataDF.loc[(dataDF['date'] == countyTime[i - 1]) & (dataDF['StateCounty'] == c)]['deaths'].values[0]
            nowDead = dataDF.loc[(dataDF['date'] == countyTime[i]) & (dataDF['StateCounty'] == c)]['deaths'].values[0]
            nextDead = dataDF.loc[(dataDF['date'] == countyTime[i + 1]) & (dataDF['StateCounty'] == c)]['deaths'].values[0]            
            if nowCase < lastCase:
                nowCase = max(int(0.5 * (lastCase + nextCase)), lastCase)
                dataDF.loc[(dataDF['date'] == countyTime[i]) & (dataDF['StateCounty'] == c), 'cases'] = nowCase
            if nowDead < lastDead:
                nowDead = max(int(0.5 * (lastDead + nextDead)), lastDead)
                dataDF.loc[(dataDF['date'] == countyTime[i]) & (dataDF['StateCounty'] == c), 'deaths'] = nowDead
        elif i == len(countyTime) - 1:
            lastCase = dataDF.loc[(dataDF['date'] == countyTime[i - 1]) & (dataDF['StateCounty'] == c)]['cases'].values[0]
            nowCase = dataDF.loc[(dataDF['date'] == countyTime[i]) & (dataDF['StateCounty'] == c)]['cases'].values[0]
            
            lastDead = dataDF.loc[(dataDF['date'] == countyTime[i - 1]) & (dataDF['StateCounty'] == c)]['deaths'].values[0]
            nowDead = dataDF.loc[(dataDF['date'] == countyTime[i]) & (dataDF['StateCounty'] == c)]['deaths'].values[0]
            if nowCase < lastCase:
                nowCase = lastCase
                dataDF.loc[(dataDF['date'] == countyTime[i]) & (dataDF['StateCounty'] == c), 'cases'] = nowCase
            if nowDead < lastDead:
                nowDead = lastDead
                dataDF.loc[(dataDF['date'] == countyTime[i]) & (dataDF['StateCounty'] == c), 'deaths'] = nowDead
    
dataDF.drop(['StateCounty'], axis = 1, inplace = True)    
dataDF.to_csv('us-counties-smooth.csv', encoding = 'utf_8_sig', index = False)

###################################################
########## combine with healthy factor ############
###################################################
factorDF = pd.read_csv('USA_factor_county.csv')
factorDF['Presence of Water Violation'] = factorDF['Presence of Water Violation'].map(lambda x: 1 if x == 'Yes' else (0 if x == 'No' else x))
factorDF.fillna(factorDF.mean(), inplace = True)

# unified the county name
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

##########
# 统计County出现以及出现次数，有很多并没有对应数据
StateCounty = dataDF.apply(lambda x: x['state'] + ' ' + x['county'], axis = 1)
dataDF.insert(0, 'StateCounty', StateCounty)

StateCounty = factorDF.apply(lambda x: x['State'] + ' ' + x['County'], axis = 1)
factorDF.insert(0, 'StateCounty', StateCounty)

stateDF = dataDF.StateCounty.value_counts()
stateDF = pd.DataFrame(stateDF).reset_index()
stateDF.rename(columns = {'index':'StateCounty', 'StateCounty':'count'}, inplace = True)
stateDF = stateDF[stateDF.StateCounty.isin(factorDF.StateCounty)]
stateDF.reset_index(drop = True, inplace = True)

#########
dataDF.drop(['fips'], axis = 1, inplace = True)
dataDF = dataDF[dataDF['StateCounty'].isin(factorDF['StateCounty'])]
colnames = dataDF.columns.values.tolist()

tempTime = dataDF['date'].unique()
tempTime.sort()
dataDF.reset_index(inplace = True, drop = True)

for i in range(len(tempTime)):
    temp1 = dataDF[dataDF['date'] == tempTime[i]]
    for j in range(stateDF.shape[0]):
        if stateDF.loc[j, 'StateCounty'] not in temp1['StateCounty'].tolist():
            temp2 = factorDF[factorDF['StateCounty'] == stateDF.loc[j, 'StateCounty']]
            temp2.reset_index(drop = True, inplace = True)
            temp3 = pd.DataFrame([[stateDF.loc[j, 'StateCounty'], tempTime[i], temp2.loc[0, 'County'], temp2.loc[0, 'State'], 0, 0]], columns = colnames)
            dataDF = dataDF.append(temp3, ignore_index = True)

##################################
# 开始处理想要得到的数据
# 每条数据包含此前7天的数据
resultDF = dataDF[dataDF['date'].isin(tempTime[:-7])]
resultDF.rename(columns = {'cases':'confirm_1', 'deaths': 'dead_1'},
                inplace = True)

resultDF['confirm_2'], resultDF['dead_2'] = None, None
resultDF['confirm_3'], resultDF['dead_3'] = None, None
resultDF['confirm_4'], resultDF['dead_4'] = None, None
resultDF['confirm_5'], resultDF['dead_5'] = None, None
resultDF['confirm_6'], resultDF['dead_6'] = None, None
resultDF['confirm_7'], resultDF['dead_7'] = None, None

nextDF = dataDF[dataDF['date'].isin(tempTime[1:-6])]
nextDF = nextDF.sort_values(by = ['date', 'StateCounty'], axis = 0)
resultDF = resultDF.sort_values(by = ['date', 'StateCounty'], axis = 0)
nextDF.reset_index(inplace = True, drop = True)
resultDF.reset_index(inplace = True, drop = True)
temp = nextDF[(nextDF['date'] == resultDF['date'] + timedelta(1)) & \
         (nextDF['StateCounty'] == resultDF['StateCounty'])]
resultDF['confirm_2'], resultDF['dead_2'] = \
    temp['cases'], temp['deaths']
    
nextDF = dataDF[dataDF['date'].isin(tempTime[2:-5])]
nextDF = nextDF.sort_values(by = ['date', 'StateCounty'], axis = 0)
resultDF = resultDF.sort_values(by = ['date', 'StateCounty'], axis = 0)
nextDF.reset_index(inplace = True, drop = True)
resultDF.reset_index(inplace = True, drop = True)
temp = nextDF[(nextDF['date'] == resultDF['date'] + timedelta(2)) & \
         (nextDF['StateCounty'] == resultDF['StateCounty'])]
resultDF['confirm_3'], resultDF['dead_3'] = \
    temp['cases'], temp['deaths']
    
nextDF = dataDF[dataDF['date'].isin(tempTime[3:-4])]
nextDF = nextDF.sort_values(by = ['date', 'StateCounty'], axis = 0)
resultDF = resultDF.sort_values(by = ['date', 'StateCounty'], axis = 0)
nextDF.reset_index(inplace = True, drop = True)
resultDF.reset_index(inplace = True, drop = True)
temp = nextDF[(nextDF['date'] == resultDF['date'] + timedelta(3)) & \
         (nextDF['StateCounty'] == resultDF['StateCounty'])]
resultDF['confirm_4'], resultDF['dead_4'] = \
    temp['cases'], temp['deaths']
    
nextDF = dataDF[dataDF['date'].isin(tempTime[4:-3])]
nextDF = nextDF.sort_values(by = ['date', 'StateCounty'], axis = 0)
resultDF = resultDF.sort_values(by = ['date', 'StateCounty'], axis = 0)
nextDF.reset_index(inplace = True, drop = True)
resultDF.reset_index(inplace = True, drop = True)
temp = nextDF[(nextDF['date'] == resultDF['date'] + timedelta(4)) & \
         (nextDF['StateCounty'] == resultDF['StateCounty'])]
resultDF['confirm_5'], resultDF['dead_5'] = \
    temp['cases'], temp['deaths']
    
nextDF = dataDF[dataDF['date'].isin(tempTime[5:-2])]
nextDF = nextDF.sort_values(by = ['date', 'StateCounty'], axis = 0)
resultDF = resultDF.sort_values(by = ['date', 'StateCounty'], axis = 0)
nextDF.reset_index(inplace = True, drop = True)
resultDF.reset_index(inplace = True, drop = True)
temp = nextDF[(nextDF['date'] == resultDF['date'] + timedelta(5)) & \
         (nextDF['StateCounty'] == resultDF['StateCounty'])]
resultDF['confirm_6'], resultDF['dead_6'] = \
    temp['cases'], temp['deaths']
    
nextDF = dataDF[dataDF['date'].isin(tempTime[6:-1])]
nextDF = nextDF.sort_values(by = ['date', 'StateCounty'], axis = 0)
resultDF = resultDF.sort_values(by = ['date', 'StateCounty'], axis = 0)
nextDF.reset_index(inplace = True, drop = True)
resultDF.reset_index(inplace = True, drop = True)
temp = nextDF[(nextDF['date'] == resultDF['date'] + timedelta(6)) & \
         (nextDF['StateCounty'] == resultDF['StateCounty'])]
resultDF['confirm_7'], resultDF['dead_7'] = \
    temp['cases'], temp['deaths']

del temp

###########
resultDF = pd.merge(resultDF, factorDF, left_on = ['StateCounty'], right_on = ['StateCounty'])
resultDF = resultDF.sort_values(by = ['date'], axis = 0)
resultDF.reset_index(inplace = True, drop = True)
resultDF.drop(['State'], axis = 1, inplace = True)

#####################
# 加上当日数据
nextDF = dataDF[dataDF['date'].isin(tempTime[7:])]
nextDF = nextDF.sort_values(by = ['date', 'StateCounty'], axis = 0)
resultDF = resultDF.sort_values(by = ['date', 'StateCounty'], axis = 0)
nextDF.reset_index(inplace = True, drop = True)
resultDF.reset_index(inplace = True, drop = True)

resultDF['cum_confirm'], resultDF['cum_dead'] = None, None

temp = nextDF[(nextDF['date'] == resultDF['date'] + timedelta(7)) & \
         (nextDF['StateCounty'] == resultDF['StateCounty'])]
resultDF['cum_confirm'], resultDF['cum_dead'] = \
    temp['cases'], temp['deaths']
   
# 如果前7天里有2天以上都没有数据，删除那一行
resultDF = resultDF[~((resultDF['confirm_1'] == 0) & (resultDF['confirm_2'] == 0))]
resultDF.drop(['County'], axis = 1, inplace = True)
resultDF.loc[:, 'date'] = nextDF.loc[:, 'date']

resultDF.to_csv('USA_data_county_smooth.csv', encoding = 'utf_8_sig', index = False)

##########
## data without cov
factorColname = factorDF.columns.values.tolist()
factorColname = factorColname[3:]
resultNoDF = resultDF.drop(factorColname, axis = 1)
resultNoDF.to_csv('USA_data_county_smooth_no.csv', encoding = 'utf_8_sig', index = False)
