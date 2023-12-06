# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

path = ".\\data\\"
seed = [2022, 2023]
ii = 0
target = [0, 1]
true = np.load(path + "true_" + str(0) + '_'+str(seed[ii]) + "_" + str(target[ii]) + ".npy", allow_pickle = True)
mae_ = []
mre_ = []
for index_ in range(6):
    pre30 = np.load(path + "pre30_" + str(index_) + '_'+str(seed[ii]) + "_" + str(target[ii]) + ".npy", allow_pickle = True)
    mre = np.abs(pre30[[15*c + 1 for c in range(10)], -1, 0]- true[[15*c + 1 for c in range(10)], 50+index_])/true[[15*c + 1 for c in range(10)], 50+index_]
    mre_.append(mre)
    
mre_ = np.array(mre_)


valDF = pd.read_csv(path + "valDF" + str(seed[ii]) + ".csv", parse_dates = ['date'] ,infer_datetime_format=True)
valpre = np.load(path + "valPredict_model_" + str(seed[ii]) + ".npy")
date = np.unique(valDF['date'])
valDF = valDF[['StateCounty', 'cum_confirm', 'date']]
top10_statecounty = ['Florida Miami-Dade', 'Louisiana Jefferson', 'Connecticut Fairfield', 'California Los Angeles', 'Michigan Wayne', 'Pennsylvania Philadelphia', 'Illinois Cook', 'Massachusetts Middlesex', 'New Jersey Bergen', 'New York New York City']
valDF['pre'] = valpre[0:len(valDF), 0]
top10_metric = pd.DataFrame([])
top10_metric['StateCounty'] = top10_statecounty
mre = []
for i in range(10):
    val_temp = valDF[valDF['StateCounty'] == top10_statecounty[i]]
    mre.append(np.mean(np.abs(val_temp['pre'] - val_temp['cum_confirm'])/val_temp['cum_confirm']))
top10_metric['MRE7'] = mre
top10_metric['MRE30'] = np.mean(mre_, axis = 0)
print(top10_metric)
    