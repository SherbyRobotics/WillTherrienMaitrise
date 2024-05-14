#!/usr/bin/env python3
import pandas as pd # data processing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as MAE
# from rich.console import Console
# from rich.table import Table
import numpy as np


# training_source =  'merged_data_no_tests.xlsx'
# training_source =  'merged_data_no_tests.xlsx'
# training_source = 'racecar_data.xlsx'
training_source = 'limo_data.xlsx'
# training_source = 'limo_racecar_data.xlsx'

data = pd.read_excel( training_source, index_col=False, engine='openpyxl')

inputs_columns = ["v_i","l","a","delta","mu","Nf","Nr","g"]
outputs_columns = ["Y"]

L = data[["l"]].values
X = data[inputs_columns].values
Y = data[outputs_columns].values

x_aug1 = X[:,5] * X[:,4] * X[:,7] / ((X[:,5]+X[:,6]) * X[:,2] )
x_aug2 = X[:,2] * X[:,4] * X[:,7] / ( X[:,0]**2 * np.tan(X[:,3]) )

X_aug1 = np.append(X,x_aug1[:,np.newaxis], 1)
X_aug2 = np.append(X_aug1,x_aug2[:,np.newaxis], 1)


model = XGBRegressor()

model.fit(X_aug2,Y)

## Test on training data
Y_hat = model.predict(X_aug2)

average_error = MAE(Y_hat,Y)
print("Self Average error : ", average_error, '[m]')

Y_normalized = Y.squeeze() / L.squeeze()
Y_hat_normalized = Y_hat / L.squeeze()
average_error_normalized = MAE(Y_hat_normalized,Y_normalized)
print("Self Average error normalized: ", average_error_normalized * 100, '%')


# Test on Xmaxx data

test_source = 'Xmaxx_test_data.xlsx'

data_test = pd.read_excel( test_source, index_col=False, engine='openpyxl')

L_test = data_test[["l"]].values
X_test = data_test[inputs_columns].values
Y_test = data_test[outputs_columns].values


x_test_aug1 = X_test[:,5] * X_test[:,4] * X_test[:,7] / ((X_test[:,5]+X_test[:,6]) * X_test[:,2] )
x_test_aug2 = X_test[:,2] * X_test[:,4] * X_test[:,7] / ( X_test[:,0]**2 * np.tan(X_test[:,3]) )

X_test_aug1 = np.append(X_test,x_test_aug1[:,np.newaxis], 1)
X_test_aug2 = np.append(X_test_aug1,x_test_aug2[:,np.newaxis], 1)

Y_hat_test = model.predict(X_test_aug2)

test_average_error = MAE(Y_hat_test,Y_test)
print("Test Average error : ", test_average_error, '[m]')

Y_test_normalized = Y_test.squeeze() / L_test.squeeze()
Y_hat_test_normalized = Y_hat_test / L_test.squeeze()
test_average_error_normalized = MAE(Y_hat_test_normalized,Y_test_normalized)
print("Test Average error normalized: ", test_average_error_normalized * 100, '%')