#!/usr/bin/env python3
import pandas as pd # data processing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as MAE
# from rich.console import Console
# from rich.table import Table
import numpy as np


training_source = 'racecar_data.xlsx'
# training_source = 'limo_data.xlsx'

data = pd.read_excel( training_source, index_col=False, engine='openpyxl')

inputs_columns = ["v_i","l","a","delta","mu","Nf","Nr","g"]
outputs_columns = ["Y"]


L = data[["l"]].values
X = data[inputs_columns].values
Y = data[outputs_columns].values

# Normalize the data with the maximum values
v_i_max = X[:,0].max()
l_max = X[:,1].max()
a_max = X[:,2].max()
delta_max = X[:,3].max()
mu_max = X[:,4].max()
nf_max = X[:,5].max()
nr_max = X[:,6].max()
g_max = X[:,7].max()
Y_max = Y.max()

X_normalized = X.copy()
X_normalized[:,0] = X[:,0] / v_i_max
X_normalized[:,1] = X[:,1] / l_max
X_normalized[:,2] = X[:,2] / a_max
X_normalized[:,3] = X[:,3] / delta_max
X_normalized[:,4] = X[:,4] / mu_max
X_normalized[:,5] = X[:,5] / nf_max
X_normalized[:,6] = X[:,6] / nr_max
X_normalized[:,7] = X[:,7] / g_max

Y_normalized = Y[:,0] / Y_max

model = XGBRegressor()

model.fit(X_normalized,Y_normalized)

# Test on training data
Y_hat_normalized = model.predict(X_normalized)
Y_hat = Y_hat_normalized * Y_max

average_error = MAE(Y_hat,Y)
print("Self Average error : ", average_error, '[m]')

Y_normalized = Y.squeeze() / L.squeeze()
Y_hat_normalized = Y_hat / L.squeeze()
average_error_normalized = MAE(Y_hat_normalized,Y_normalized)
print("Self Average error normalized: ", average_error_normalized * 100, '%')


# Test data

test_source = 'Xmaxx_data.xlsx'

data_test = pd.read_excel( test_source, index_col=False, engine='openpyxl')

L_test = data_test[["l"]].values
X_test = data_test[inputs_columns].values
Y_test = data_test[outputs_columns].values


# Predictions
X_test_normalized = X_test.copy()
X_test_normalized[:,0] = X_test[:,0] / v_i_max
X_test_normalized[:,1] = X_test[:,1] / l_max
X_test_normalized[:,2] = X_test[:,2] / a_max
X_test_normalized[:,3] = X_test[:,3] / delta_max
X_test_normalized[:,4] = X_test[:,4] / mu_max
X_test_normalized[:,5] = X_test[:,5] / nf_max
X_test_normalized[:,6] = X_test[:,6] / nr_max
X_test_normalized[:,7] = X_test[:,7] / g_max


Y_hat_test_normalized = model.predict(X_test_normalized)

Y_hat_test = Y_hat_test_normalized * Y_max


# Evaluation
test_average_error = MAE(Y_hat_test,Y_test)
print("Test Average error : ", test_average_error, '[m]')

Y_test_normalized = Y_test.squeeze() / L_test.squeeze()
Y_hat_test_normalized = Y_hat_test / L_test.squeeze()
test_average_error_normalized = MAE(Y_hat_test_normalized,Y_test_normalized)
print("Test Average error normalized: ", test_average_error_normalized * 100, '%')