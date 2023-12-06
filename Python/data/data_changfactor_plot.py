# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 12:35:16 2023

@author: Wenting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
ticker_spacing = 7
target = [0, 1]
seed = [24, 23]
path = "~.\\data\\"
fig = plt.figure(figsize=(12,10), dpi = 300)
ax = fig.subplots(nrows=3, ncols=2)
title = 'abc' 
fig.subplots_adjust(hspace= 0.3)
index = [1,2,0]
index_ = 0
for ii in [0, 1]:
    date = np.load(path + "date_" + str(index_) + '_'+ str(seed[ii]) + "_" + str(target[ii]) + ".npy", allow_pickle = True)
    pre30 = np.load(path + "pre30_" + str(index_) + '_'+str(seed[ii]) + "_" + str(target[ii]) + ".npy", allow_pickle = True)
    cflist = np.load(path + "cflist_" + str(index_) + '_'+str(seed[ii]) + "_" + str(target[ii]) + ".npy", allow_pickle = True)
    cfilist = np.load(path + "cfilist_" + str(index_) + '_'+str(seed[ii]) + "_" + str(target[ii]) + ".npy", allow_pickle = True)
    true = np.load(path + "true_" + str(index_) + '_'+str(seed[ii]) + "_" + str(target[ii]) + ".npy", allow_pickle = True)

    fc = ['Traffic volume', 'Severe housing problems', 'Air pollution particulate matter ']
    colors = ['#85A5BC', '#87D4B2', '#FEF27D', 'orange', '#90679A']
    colors2 = ['#85A5BC', '#87D4B2', '#FEF27D', '#90679A', 'white']
    label = ['4', '3', '2', None, '0.5']

    county =  ['Florida Miami-Dade', 'Louisiana Jefferson', 'Connecticut Hartford', 'California Los Angeles', 'Michigan Wayne', 'Pennsylvania Philadelphia', 'Illinois Cook', 'Massachusetts Middlesex', 'New Jersey Essex', 'New York New York City']

    start = 28
    c = 3
    true_ = true[15*c]
    d = date[15*c, :51]
    for i in range(3):
        pre_1 = pre30[15*c + 5*i + 1, :, 0]
        
        for j in range(5):
            y = pre30[15*c + 5*index[i] + 4 - j, start:, 0]
            ax[i][ii].plot(d[start:-7], y[:-7], label = label[j], color = colors[j], linewidth = 1)
            ax[i][ii].set_ylim([min(y[:-7]), max(y[:-7])*1.013])
            ax[i][ii].fill_between(d[start:-7], y[:-7], facecolor=colors2[j], interpolate=True)
            ax[i][ii].get_yaxis().get_major_formatter().set_scientific(False)
            ax[i][ii].xaxis.set_major_locator(ticker.MultipleLocator(ticker_spacing))
            
            ax[i][ii].set_xticklabels(d[start:-1:7], 
                                     rotation = 0, 
                                     fontsize = 'medium')
            ax[i][ii].set_title(title[i] + "."+ fc[i] + "(" + county[c] + ")", loc = "left", fontsize = 'medium')
            if ii:
                ax[i][ii].set_ylabel('Confirmed Deaths')
            else:
                ax[i][ii].set_ylabel('Confirmed Cases')
        #x_major_locator = MultipleLocator(10)
        #ax_ = plt.gca()
        #ax_.xaxis.set_major_locator(x_major_locator)
lines, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(lines, 
           labels, 
           ncol=5,
           loc = 'upper center', 
           fontsize = 16,
           #mode = "expand",
           handlelength = 2,
           handleheight = 1)
fig.savefig(path +county[c] + '.pdf')