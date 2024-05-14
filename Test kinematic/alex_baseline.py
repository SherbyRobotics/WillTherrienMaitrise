#!/usr/bin/env python3
import pandas as pd # data processing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as MAE
# from rich.console import Console
# from rich.table import Table
import numpy as np


# training_source =  'merged.xlsx'
# training_source = 'Xmaxx.xlsx'
# training_source = 'racecar.xlsx'
training_source = 'limo.xlsx'

data = pd.read_excel( training_source, index_col=False, engine='openpyxl')

#All columb = v_i	l	a	delta	pi1	pi2	pi3	pi4	pi5	X	Y	theta	pi6

inputs_columns = ["v_i","l","a","delta"]
outputs_columns = ["Y"]

L = data[["l"]].values
X = data[inputs_columns].values
Y = data[outputs_columns].values

model = XGBRegressor()

model.fit(X,Y)

## Test on training data
Y_hat = model.predict(X)

average_error = MAE(Y_hat,Y)
print("Self Average error : ", average_error, '[m]')

Y_normalized = Y.squeeze() / L.squeeze()
Y_hat_normalized = Y_hat / L.squeeze()
average_error_normalized = MAE(Y_hat_normalized,Y_normalized)
print("Self Average error normalized: ", average_error_normalized * 100, '%')


# Test on Xmaxx data

test_source = 'Xmaxx.xlsx'

data_test = pd.read_excel( test_source, index_col=False, engine='openpyxl')

L_test = data_test[["l"]].values
X_test = data_test[inputs_columns].values
Y_test = data_test[outputs_columns].values

Y_hat_test = model.predict(X_test)

test_average_error = MAE(Y_hat_test,Y_test)
print("Test Average error : ", test_average_error, '[m]')

Y_test_normalized = Y_test.squeeze() / L_test.squeeze()
Y_hat_test_normalized = Y_hat_test / L_test.squeeze()
test_average_error_normalized = MAE(Y_hat_test_normalized,Y_test_normalized)
print("Test Average error normalized: ", test_average_error_normalized * 100, '%')