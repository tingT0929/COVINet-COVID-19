# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
ticker_spacing = 44
target = [0, 1]
seed = [24, 23]
path = ".\\data\\"
fig = plt.figure(figsize=(15,10), dpi = 300)
ax = fig.subplots(nrows=5, ncols=2)
fig.subplots_adjust(hspace= 0.3)
ii = 1
t = ii
valDF = pd.read_csv(path + "valDF" + str(seed[ii]) +".csv", parse_dates = ['date'] ,infer_datetime_format=True)

county =  ['Florida Miami-Dade', 'Louisiana Jefferson', 'California Los Angeles', 'Michigan Wayne', 'New York New York City','Pennsylvania Philadelphia', 'Illinois Cook', 'Massachusetts Middlesex', 'New Jersey Bergen', 'Connecticut Fairfield']

pre = np.load(path + "valPredict_model_" + str(seed[ii]) + ".npy")

fc = ['Average Daily PM2.5', 'Average Traffic Volume per Meter of Major Roadways', 'Severe Housing Problem']
valDF['pre'] = pre[:, 0]
colors = ['#9cb7de', '#e49854']
label = ['Actual', 'Predicted']
date_ = np.load("date" + str(target[ii]) + ".npy", allow_pickle = True)
cum = np.load("cum" + str(target[ii]) + ".npy", allow_pickle = True)
sc = np.load("state" + str(target[ii]) + ".npy", allow_pickle = True)
df = pd.DataFrame([])
df['date'] = date_
df['cum'] = cum
df['sc'] = sc
start = 1000
for c in range(10):
    df_temp = valDF[valDF['StateCounty'] == county[c]]
    df_temp2 = df[df['sc'] == county[c]]
    if c >= 5:
        i = c-5
        ii = 0
    else:
        i = c
        ii = 1
    if t:
        true_ = df_temp['cum_dead']
    else:
        true_ = df_temp['cum_confirm']
    d = df_temp['date']
    d = [str(i).split(' ')[0] for i in d]
    d2 = np.concatenate([df_temp2['date'], d])
    true2 = np.concatenate([df_temp2['cum'], true_])
    y = df_temp['pre']
    print(county[c])
    #print(np.mean(np.abs(y -  true_)))
    #print(np.mean(np.abs(y -  true_)/true_))
    y = np.array(y)
    true_ = np.array(true_)
    print(np.abs(y[-1] -  true_[-1])/true_[-1])
    ax[i][ii].plot(d2[-270:], true2[-270:], label = label[0], color = colors[0], linewidth = 4)
    y = np.insert(np.array(y), 0, true2[-(len(d) + 1)])
    ax[i][ii].plot(d2[-(len(d) + 1):], y, label = label[1], color = colors[1], linewidth = 2)
    ax[i][ii].set_ylim([min(min(y), min(true2[-270:]))*0.8, max(max(y), max(true2[-100:]))*1.02])
    
    ax[i][ii].xaxis.set_major_locator(ticker.MultipleLocator(ticker_spacing))
    ax[i][ii].get_yaxis().get_major_formatter().set_scientific(False)
    ax[i][ii].set_xticklabels(d2[-270:-1:ticker_spacing], 
                                     rotation = 10, 
                                     fontsize = 'medium')
    ax[i][ii].set_yticklabels(ax[i][ii].get_yticklabels(), fontsize= 'medium')
    ax[i][ii].set_title(county[c], loc = "left", fontsize = 'x-large')
    if t:
        ax[i][ii].set_ylabel('Confirmed Deaths', fontsize = 'large')
    else:
        ax[i][ii].set_ylabel('Confirmed Cases', fontsize = 'large')

lines, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(lines, 
           labels, 
           ncol=2,
           loc = 'upper center', 
           fontsize = 16,
           handlelength = 2,
           handleheight = 1)
if t:
    fig.savefig(path +'dead' + '.pdf')
else:
    fig.savefig(path +'case' + '.pdf')