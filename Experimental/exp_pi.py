#!/usr/bin/env python3
import pandas as pd # data processing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as MAE
import numpy as np


# training_source =  'merged_data_no_tests.xlsx'
# training_source = 'merged_data.xlsx'
# training_source = 'Xmaxx.xlsx'
# training_source = 'racecar_data.xlsx'
training_source = 'limo_data.xlsx'
# training_source = 'limo_racecar_data.xlsx'

data = pd.read_excel( training_source, index_col=False, engine='openpyxl')

#All columb = v_i	l	a	delta	pi1	pi2	pi3	pi4	pi5	X	Y	theta	pi6

inputs_columns = ["v_i","l","a","delta","mu","Nf","Nr","g"]
outputs_columns = ["Y"]

L = data[["l"]].values
X = data[inputs_columns].values
Y = data[outputs_columns].values

# Pi
X_pi = np.zeros((X.shape[0],5))
Y_pi = np.zeros((X.shape[0],1))

X_pi[:,0] = X[:,2] * X[:,1] / X[:,0]**2   # a * l / v_i**2
X_pi[:,1] = X[:,3] # delta 
X_pi[:,2] = X[:,5] / X[:,6] # Nf/Nr
X_pi[:,3] = X[:,4] # mu
X_pi[:,4] = X[:,7] * X[:,1] / X[:,0]**2 # 
Y_pi[:,0] = Y[:,0] / X[:,1]

model = XGBRegressor()

model.fit(X_pi,Y_pi)

# Test on training data
Y_hat_pi = model.predict(X_pi)
Y_hat = Y_hat_pi * X[:,1]

average_error = MAE(Y_hat,Y)
print("Self Average error : ", average_error, '[m]')

Y_normalized = Y.squeeze() / L.squeeze()
Y_hat_normalized = Y_hat / L.squeeze()
average_error_normalized = MAE(Y_hat_normalized,Y_normalized)
print("Self Average error normalized: ", average_error_normalized * 100, '%')


# Test data

test_source = 'Xmaxx_test_data.xlsx'

data_test = pd.read_excel( test_source, index_col=False, engine='openpyxl')

L_test = data_test[["l"]].values
X_test = data_test[inputs_columns].values
Y_test = data_test[outputs_columns].values


# Predictions
# Pi
X_test_pi = np.zeros((X_test.shape[0],5))
Y_test_pi = np.zeros((X_test.shape[0],1))

X_test_pi[:,0] = X_test[:,2] * X_test[:,1] / X_test[:,0]**2   # a * l / v_i**2
X_test_pi[:,1] = X_test[:,3] # delta 
X_test_pi[:,2] = X_test[:,5] / X_test[:,6] # Nf/Nr
X_test_pi[:,3] = X_test[:,4] # mu
X_test_pi[:,4] = X_test[:,7] * X_test[:,1] / X_test[:,0]**2 # 
# X_test_pi[:,2] = X_test[:,4] / X_test[:,5] # Nf/Nr
# X_test_pi[:,3] = X_test[:,6] * X_test[:,1] / X_test[:,0]**2 # delta 
Y_test_pi[:,0] = Y_test[:,0] / X_test[:,1]

# Test on training data
Y_hat_test_pi = model.predict(X_test_pi)
Y_hat_test = Y_hat_test_pi * X_test[:,1]


# Evaluation
test_average_error = MAE(Y_hat_test,Y_test)
print("Test Average error : ", test_average_error, '[m]')

Y_test_normalized = Y_test.squeeze() / L_test.squeeze()
Y_hat_test_normalized = Y_hat_test / L_test.squeeze()
test_average_error_normalized = MAE(Y_hat_test_normalized,Y_test_normalized)
print("Test Average error normalized: ", test_average_error_normalized * 100, '%')