# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 12:34:12 2018

@author: jesusfbes
"""


import scipy.io as sio
import numpy as np

import matplotlib.pyplot as plt

import GPy

from gp_ard import gp_ard

# SETTINGS
GPy.plotting.change_plotting_library('matplotlib')

# PARAMS
path_data = 'data/dataMatrixOsc.mat'
path_results = "results/ARD_values.mat"

labels = ['fks', 'fkr', 'fto', 'fcal', 'fk1', 'fna', 'fnaca', 'fnak']

N_fold = 10


aux_data = sio.loadmat(path_data)

# mean_APD_data = sio.loadmat('G:\Mi unidad\Investigacion\Paper_2_ISO\JFB\data\mean_APD.mat')

# APD_mean = np.concatenate( (mean_APD_data["APD_baseline_ave"].T ,mean_APD_data["APD_SP_ave"].T)).T


X = aux_data["b_Reducido"]
#Y_matrix = aux_data["Matriz_STC"].T

# sd_index = 0
#sdf_index = 2

#nsd_index = 4
# nsdf_index = 6

#stv_index = 8
#stvf_index = 10

#plf_index = 16
#nplf_index = 22


sd_low = aux_data["sd_low"]
sd_high = aux_data["sd_high"]

nsd_low = aux_data["nsd_low"]
nsd_high = aux_data["nsd_high"]

stv_low = aux_data["stv_low"]
stv_high = aux_data["stv_high"]

nstv_low = aux_data["nstv_low"]
nstv_high = aux_data["nstv_high"]

plf_low = aux_data["plf_low"]
plf_high = aux_data["plf_high"]

nplf_low = aux_data["nplf_low"]
nplf_high = aux_data["nplf_high"]


X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
# print(np.mean(X,axis=0))

n_vars = X.shape[1]

print(n_vars)

prev_results = []
# sio.loadmat(path_results)

# """
# DELTA THINGS
#
# """

label1 = "change"
label2 = "sd"
y = sd_high - sd_low  # ISO high - low
y = (y - np.mean(y)) / np.std(y)

ard_values, sign_values, mean_sq_error = gp_ard(X, y)

print("Average MSEs =" + str((np.mean(mean_sq_error))))

plt.bar(range(n_vars), ard_values.flatten())
plt.rc('text')
plt.rc('font', family='serif')
plt.rc('font', size=14)
plt.title(label1 + " " + label2)
plt.xticks(range(n_vars), labels)
#plt.title('Params. relevant for SD\n for ' + label + " conditions ")
# plt.savefig("ard_" + label1 + "_" + label2 + ".png")

plt.show()

prev_results['ARD_' + label1 + "_" + label2] = ard_values
prev_results["labels"] = labels
prev_results["sign_values_" + label1 + "_" + label2] = sign_values

sio.savemat(path_results, prev_results)


# y = nsd_high - nsd_low  # ISO high - low
# y = (y - np.mean(y)) / np.std(y)
# gp_ard(X, y, "change", "nsd")


# y = stv_high - stv_low  # ISO high - low
# y = (y - np.mean(y)) / np.std(y)
# gp_ard(X, y, "change", "stv")

# y = nstv_high - nstv_low  # ISO high - low
# y = (y - np.mean(y)) / np.std(y)
# gp_ard(X, y, "change", "nstv")

# y = plf_high - plf_low  # ISO high - low
# y = (y - np.mean(y)) / np.std(y)
# gp_ard(X, y, "change", "plf")

# y = nplf_high - nplf_low  # ISO high - low
# y = (y - np.mean(y)) / np.std(y)
# gp_ard(X, y, "change", "nplf")
# #
# #
# # """
# # ISO LOW
# #
# # """
# y = sd_low  # ISO low
# y = (y - np.mean(y)) / np.std(y)
# gp_ard(X, y, "low", "sd")

# y = nsd_low  # ISO  low
# y = (y - np.mean(y)) / np.std(y)
# gp_ard(X, y, "low", "nsd")


# y = stv_low  # ISO  low
# y = (y - np.mean(y)) / np.std(y)
# gp_ard(X, y, "low", "stv")

# y = nstv_low  # ISO low
# y = (y - np.mean(y)) / np.std(y)
# gp_ard(X, y, "low", "nstv")

# y = plf_low  # ISO low
# y = (y - np.mean(y)) / np.std(y)
# gp_ard(X, y, "low", "plf")

# y = nplf_low  # ISO low
# y = (y - np.mean(y)) / np.std(y)
# gp_ard(X, y, "low", "nplf")

# #
# #
# # """
# # ISO HIGH
# #
# # """
# y = sd_high  # ISO high
# y = (y - np.mean(y)) / np.std(y)
# gp_ard(X, y, "high", "sd")

# y = nsd_high  # ISO high
# y = (y - np.mean(y)) / np.std(y)
# gp_ard(X, y, "high", "nsd")

# y = stv_high  # ISO  high
# y = (y - np.mean(y)) / np.std(y)
# gp_ard(X, y, "high", "stv")

# y = nstv_high  # ISO high
# y = (y - np.mean(y)) / np.std(y)
# gp_ard(X, y, "high", "nstv")

# y = plf_high  # ISO high
# y = (y - np.mean(y)) / np.std(y)
# gp_ard(X, y, "high", "plf")

# y = nplf_high  # ISO high
# y = (y - np.mean(y)) / np.std(y)
# gp_ard(X, y, "high", "nplf")
