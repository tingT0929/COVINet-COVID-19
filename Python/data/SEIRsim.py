# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random



random.seed(2024)

days_per_year = 365
num_days = days_per_year * 2
t = np.linspace(0, num_days, num_days)


def deriv(y, t, N, beta, sigma, gamma, coe, cov):
    S, E, I, R = y
    
    dSdt = -(beta * S * I ) / N 
    
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    
    return dSdt+ coe * cov, dEdt, dIdt, dRdt

cov_1 = []
cov_2 = []
cov_3 = []
beta_list = [0.4, 0.3, 0.7]
sigma_list = [0.015, 0.015, 0.015]
coe_list = [1.0, 0.1]
random.seed(1)
cov1_list = [random.uniform(0, 1) for i in range(100)]
random.seed(2)
cov2_list = [random.uniform(0, 1) for i in range(100)]
random.seed(3)
cov3_list = [random.uniform(0, 1) for i in range(100)]
random_list = [random.uniform(-0.001, 0.001) for i in range(100)]

for i in range(100):
    N = 1000
    I0 = 1      
    E0 = 2      
    R0 = 0       
    S0 = N - I0 - E0 - R0  
    for j in range(3):
        beta = beta_list[j] + random_list[i]  
        sigma =sigma_list[j] + random_list[i]  
        gamma = 0.25  + random_list[i]  
        
        y0 = S0, E0, I0, R0
        
        
        ret = odeint(deriv, y0, t, args=(N, beta, sigma, gamma, 0, 0))
        S, E, I, R = ret.T
        cumulative_cases = N - S 
        
        cumulative_cases += np.array(range(len(S)))*(cov1_list[i] *coe_list[0] + cov2_list[i]*coe_list[1])
        
        
        ret2 = odeint(deriv, y0, t, args=(N, beta, sigma, gamma, 0, 0))
        S2, E, I, R = ret2.T
        
        start_date = pd.Timestamp('2022-01-01')
        dates = pd.date_range(start=start_date, periods = days_per_year*2)
        
        df = pd.DataFrame([])
        df['date'] = dates
        df['county'] = i*3 + j
            
        df['case'] = cumulative_cases
           
        if i+j != 0:
                    df.to_csv(".\dfsim.csv", mode = 'a', index = False, header = False)
        else:
                    df.to_csv(".\dfsim.csv", index = False, header = True)
        cov_1.append(cov1_list[i])
        cov_2.append(cov2_list[i])
    
df = pd.DataFrame(np.array([cov_1, cov_2])).T
df.columns = ['cov_' + str(i + 1) for i in range(2)]
df['county'] = range(300)
df.to_csv(".\factorsim.csv", index = False, header = True)


