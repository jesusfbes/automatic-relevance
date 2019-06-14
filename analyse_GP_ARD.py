# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 12:34:12 2018

@author: jesusfbes
"""


import scipy.io as sio
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt

import GPy

from gp_ard import gp_ard

# SETTINGS
GPy.plotting.change_plotting_library('matplotlib')


# load data
data = pd.read_csv("sample_data.csv")

labels= list(data)[:-1]

y = np.array(data.iloc[:,-1])
X = np.array(data.iloc[:,:-1])


ard_values, sign_values, mean_sq_error = gp_ard(X, y)

# Show results
print(f"ARD values {ard_values}")
print(f"Sign values {sign_values}")

print(f"Average MSEs {np.mean(mean_sq_error)}")

n_vars = len(labels)
plt.bar(range(n_vars), ard_values.flatten())
plt.rc('text')
plt.rc('font', family='serif')
plt.rc('font', size=14)
plt.xticks(range(n_vars), labels)


plt.show()
