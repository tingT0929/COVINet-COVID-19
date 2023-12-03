import numpy as np
import pandas as pd
import math
from dbfread import DBF

latLongTable = DBF('"~./data/lat_long.dbf')
latLongDF = pd.DataFrame(iter(latLongTable))
latLongDF.drop(['STATE', 'CWA', 'TIME_ZONE', 'FE_AREA'], axis = 1, inplace = True)
latLongDF['FIPS'] = latLongDF['FIPS'].astype(np.int64)
latLongDF.rename(columns = {'FIPS': 'fips'}, inplace = True)
print(latLongDF.columns)

eg_data = pd.read_csv("~./data/USA_data_county.csv")

col = [i for i in eg_data.columns]

col.insert(18, 'confirm_8')
col.insert(19,'dead_8')
col.insert(20,'confirm_9')
col.insert(21,'dead_9')
col.insert(22,'confirm_10')
col.insert(23,'dead_10')
col.insert(24,'confirm_11')
col.insert(25,'dead_11')
col.insert(26,'confirm_12')
col.insert(27,'dead_12')
col.insert(28,'confirm_13')
col.insert(29,'dead_13')
col.insert(30,'confirm_14')
col.insert(31,'dead_14')


factor = pd.read_csv("~./data/USA_factor_county.csv")

factor['StateCounty'] = factor['State'] + ' ' + factor['County']
case = pd.read_csv("~./data/us-2023.csv")
print("data loaded...")

latLongDF.drop_duplicates(subset = ['fips'], inplace = True)
case = pd.merge(case, latLongDF, left_on = ['fips'], right_on = ['fips'], how = 'left')
print("case columns:", case.columns)
print("data loaded")
print(case['date'])
print("date selected")
case['StateCounty'] = case['state'] + ' ' + case['county']
county = np.unique(case['StateCounty'])
df = pd.merge(case, factor, how='left', on='StateCounty')
print("data merged")
df = df[df['county']!= 'Unknown']
df = df[df['state']!= 'Unknown']
print(df.columns)

df = df.drop(['fips', 'State', 'County'], axis = 1)
df['Presence of Water Violation'][df['Presence of Water Violation'] == "Yes"] = 1
df['Presence of Water Violation'][df['Presence of Water Violation'] == "No"] = 0
for i in range(len(county)):
    df_csv = df[df['StateCounty'] == county[i]]
    
    df_csv.sort_values(by='date', ascending = True)
    
    for ii in range(14):
            df_csv.insert(4, 'confirm_' + str(ii+1), 0)
            df_csv.insert(4, 'dead_' + str(ii+1), 0)
    df_csv.reset_index(drop = True, inplace = True)
    add = 0
    case = df_csv['cases']
    death = df_csv['deaths']
    case = np.array(case)
    death = np.array(death)
    for ii in range(len(case)-1):
        case[ii+1] = max(case[ii], case[ii+1])
        death[ii+1] = max(death[ii], death[ii+1])
    df_csv['cases'] = case
    df_csv['deaths'] = death
    for times in range(21):
        col_ = df_csv.columns
        
        try:
              df_csv = pd.DataFrame(np.insert(df_csv.values, 0, values=df_csv.iloc[0,:], axis=0))
        except:
              pass
        df_csv.columns = col_
    for ii  in range(21, df_csv.shape[0]):
        for jj in range(14):
                
                df_csv['confirm_' + str(14 - jj)][ii] = df_csv['cases'][ii - jj -1]
                df_csv['dead_' + str(14 - jj)][ii] = df_csv['deaths'][ii - jj -1]
    df_csv = df_csv[21:]
    df_csv = df_csv.rename(columns={'deaths':'cum_dead'})
    df_csv = df_csv.rename(columns={'cases':'cum_confirm'})
    
    df_csv = df_csv[col]
    
    df_csv = df_csv.fillna(0)
    if i == 0:
        df_csv.to_csv("~./data/data-14-new.csv", header = True, index = False)
    else:
        df_csv.to_csv("~./data/data-14-new.csv", mode='a', index = False, header = False)

    #print(str(i+1), county[i], " written")