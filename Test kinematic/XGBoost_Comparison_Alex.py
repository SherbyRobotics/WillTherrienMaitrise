#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 10:15:47 2023

@author: William Therrien
"""
import pandas as pd # data processing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as MAE
# from rich.console import Console
# from rich.table import Table
import numpy as np

def testModels(models,X_tests,y_tests, var_holder, cors_full):
    inc = 0
    MAE_list = []
    for i in range(0,len(X_tests)):
        X_test = X_tests[i] #input tests du vehicule i
        y_test = y_tests[i] #output tests du vehicule i
        cors   = cors_full[i]
        z = 0
        for model in models: #Models for all vehicles
            k = 0
            z += 1
            for model_type in model: #Types of model (either X, Y or theta)
                X_test_type = X_test[k]
                y_test_type = y_test[k]
                cor = cors[k] #Multiply wheelbase l to pi1 and pi2 if it's a dimensionless model, if not multiply by 1.
                pred = model_type.predict(X_test_type)
                if (k == 2): #If it's pi3 = theta
                    m_a_e = MAE(pred,y_test_type)
                else: #If it's pi1=X/l or pi2=Y/l, multiply by l so that the metrics are comparable
                    m_a_e = MAE(pred*cor,y_test_type*cor)
                k += 1
                MAE_list.append(m_a_e) 
                inc += 1

            var_holder['data'+str(i+1)+'_'+str(z)] = " X: "+str(round(MAE_list[inc-3],4))+"\n Y: "+str(round(MAE_list[inc-2],4))+"\n θ: "+str(round(MAE_list[inc-1],4))
    return var_holder, MAE_list

def printTable():
    
    #table = Table(show_lines=True)
#    columns = ["MODELS", "Racecar data", "Limo data","Xmaxx data","Merged data","Civic data"]
#    rows = [["\n Racecar model \n", data1_1, data2_1, data3_1, data4_1, data5_1],
#            ["\n Limo model \n",    data1_2, data2_2, data3_2, data4_2, data5_2],
#            ["\n Xmaxx model \n",   data1_3, data2_3, data3_3, data4_3, data5_3],
#            ["\n MERGED model \n",  data1_4, data2_4, data3_4, data4_4, data5_4],
#            ["\n Civic model \n",   data1_5, data2_5, data3_5, data4_5, data5_5],]
    
    columns = ["MODELS", "Racecar data", "Limo data","Xmaxx data","civic"]
    rows = [["\n Racecar model \n", data1_1, data2_1, data3_1,"N/A"],
            ["\n Limo model \n",    data1_2, data2_2, data3_2,"N/A"],
            ["\n Xmaxx model \n",   data1_3, data2_3, data3_3,"N/A"],
            ["\n MERGED model \n",  data1_4, data2_4, data3_4,data5_4],]
    
    

    for column in columns:
        # table.add_column(column)
        pass

    for row in rows:
        table.add_row(*row, style='bright_green')
        pass

    #console = Console()
    #console.print(table)    
    
    return

def printReport(MAE_list):
    print("")
    print("##################   SELF-PREDICTION   ################")
    print("MEAN MAE SELF-PREDICTION (X): "+str(round(((MAE_list[0]+MAE_list[18]+MAE_list[36])/3),4))+" m")
    print("MEAN MAE SELF-PREDICTION (Y): "+str(round(((MAE_list[1]+MAE_list[19]+MAE_list[37])/3),4))+" m")
    print("MEAN MAE SELF-PREDICTION (θ): "+str(round(((MAE_list[2]+MAE_list[20]+MAE_list[38])/3),4))+" rad")
    print("")
    print("##################   CROSS-PREDICTION  ################")
    print("MEAN MAE CROSS-PREDICTION (X): "+str(round(((MAE_list[3]+MAE_list[6]+MAE_list[15]+MAE_list[21]+MAE_list[30]+MAE_list[33])/6),4))+" m")
    print("MEAN MAE CROSS-PREDICTION (Y): "+str(round(((MAE_list[4]+MAE_list[7]+MAE_list[16]+MAE_list[22]+MAE_list[31]+MAE_list[34])/6),4))+" m")
    print("MEAN MAE CROSS-PREDICTION (θ): "+str(round(((MAE_list[5]+MAE_list[8]+MAE_list[17]+MAE_list[23]+MAE_list[32]+MAE_list[35])/6),4))+" rad")
    print("")
    print("##################   SHARED-PREDICTION ################")
    print("MEAN MAE SHARED-PREDICTION (X): "+str(round(((MAE_list[9]+MAE_list[24]+MAE_list[39])/3),4))+" m")
    print("MEAN MAE SHARED-PREDICTION (Y): "+str(round(((MAE_list[10]+MAE_list[25]+MAE_list[40])/3),4))+" m")
    print("MEAN MAE SHARED-PREDICTION (θ): "+str(round(((MAE_list[11]+MAE_list[26]+MAE_list[41])/3),4))+" rad")
    return

def createModels(excel_file, drop_data, outputs):
    
    #Choose test size (a value of 0.2 takes 20% of database as test and 80% as train)
    size = 0.2
    
    #Load data set
    data = pd.read_excel(excel_file,index_col=False, engine='openpyxl')
    
    #Remove columns in database so that the model does not train with these values
    X = data.drop(drop_data,axis=1)
    X = X.values
    list_L = data["l"].values
    
    #Set the output for all 3 models (X,Y,theta or pi1,pi2,pi3)
    y1 = data[outputs[0]].values
    y2 = data[outputs[1]].values
    y3 = data[outputs[2]].values
    
    #Create models
    model1 = XGBRegressor()
    model2 = XGBRegressor()
    model3 = XGBRegressor()
    
    #Split data
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1, test_size=size, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size=size, random_state=42)    
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y3, test_size=size, random_state=42)
    
    #Track length of wheelbase 
    fake, cor, fake ,fake = train_test_split(list_L, y1, test_size=size, random_state=42)

    #Fit models
    model1.fit(X_train1,y_train1)
    model2.fit(X_train2,y_train2)
    model3.fit(X_train3,y_train3)

    #Package models and tests
    models = [model1, model2, model3]
    X_tests = [X_test1, X_test2, X_test3]
    y_tests = [y_test1, y_test2, y_test3] 
    cors = [cor, cor, cor]      
  
    return models, X_tests, y_tests, cors


print("########################################################################################")
print("#                         MODELS BASED ON DIMENSIONNALIZED PARAMS                      #")
print("########################################################################################")
p_drop = ["pi1","pi2","pi3","pi4","pi5","pi6","X","Y","theta"]
p_outputs = ["X","Y","theta"]


# Physical dimensionnalized models for RACECAR
p_racecar_M, p_racecar_X, p_racecar_y, p_racecar_c = createModels("racecar.xlsx",p_drop, p_outputs)
# Physical dimensionnalized models for LIMO
p_limo_M, p_limo_X, p_limo_y, p_limo_c = createModels("limo.xlsx",p_drop, p_outputs)
# Physical dimensionnalized models for XMAXX
p_Xmaxx_M, p_Xmaxx_X, p_Xmaxx_y, p_Xmaxx_c = createModels("Xmaxx.xlsx",p_drop, p_outputs)
# Physical dimensionnalized models for merged files of all 3 small vehicles
p_merged_M, p_merged_X, p_merged_y, p_merged_c = createModels("merged.xlsx",p_drop, p_outputs)
# Physical dimensionnalized models for CIVIC
p_civic_M, p_civic_X, p_civic_y, p_civic_c = createModels("civic.xlsx",p_drop, p_outputs)

#Packaging
p_models      = [p_racecar_M, p_limo_M, p_Xmaxx_M, p_merged_M, p_civic_M]
p_X_tests     = [p_racecar_X, p_limo_X, p_Xmaxx_X, p_merged_X, p_civic_X]
p_y_tests     = [p_racecar_y, p_limo_y, p_Xmaxx_y, p_merged_y, p_civic_y]
p_corrections = [np.ones(len(p_racecar_c)), np.ones(len(p_limo_c)), np.ones(len(p_Xmaxx_c)), np.ones(len(p_merged_c)), np.ones(len(p_civic_c))]

# Test and print self, cross and shared predictions MAE (Mean Absolute Error)
var_holder = {}
var_holder, MAE_list = testModels(p_models, p_X_tests, p_y_tests, var_holder, p_corrections)
locals().update(var_holder)
# printTable()
printReport(MAE_list)


print("")
print("########################################################################################")
print("#                              BUCKINGHAM BASED MODELS                                 #")
print("########################################################################################")
b_drop = ["v_i","l","a","delta","pi1","pi2","pi3","pi6","X","Y","theta"]
b_outputs = ["pi1","pi2","pi3"]

# Buckingham based non-dimensionnalized models for RACECAR
b_racecar_M, b_racecar_X, b_racecar_y, b_racecar_c = createModels("racecar.xlsx",b_drop, b_outputs)
# Buckingham based non-dimensionnalized models for LIMO
b_limo_M, b_limo_X, b_limo_y, b_limo_c = createModels("limo.xlsx",b_drop, b_outputs)
# Buckingham based non-dimensionnalized models for XMAXX
b_Xmaxx_M, b_Xmaxx_X, b_Xmaxx_y, b_Xmaxx_c = createModels("Xmaxx.xlsx",b_drop, b_outputs)
# Buckingham based non-dimensionnalized models for merged files of all 3 small vehicles
b_merged_M, b_merged_X, b_merged_y, b_merged_c = createModels("merged.xlsx",b_drop, b_outputs)
# Buckingham based non-dimensionnalized models for CIVIC
b_civic_M, b_civic_X, b_civic_y, b_civic_c = createModels("civic.xlsx",b_drop, b_outputs)

#Packaging
b_models      = [b_racecar_M, b_limo_M, b_Xmaxx_M, b_merged_M, b_civic_M]
b_X_tests     = [b_racecar_X, b_limo_X, b_Xmaxx_X, b_merged_X, b_civic_X]
b_y_tests     = [b_racecar_y, b_limo_y, b_Xmaxx_y, b_merged_y, b_civic_y]
b_corrections = [b_racecar_c, b_limo_c, b_Xmaxx_c, b_merged_c, b_civic_c]

# Test and print self, cross and shared predictions MAE (Mean Absolute Error)
var_holder = {}
var_holder, MAE_list = testModels(b_models, b_X_tests, b_y_tests, var_holder, b_corrections)
locals().update(var_holder)
#printTable()
printReport(MAE_list)


print("")
print("########################################################################################")
print("#                         AUGMENTED BUCKINGHAM BASED MODELS                            #")
print("########################################################################################")
a_drop = ["v_i","l","a","delta","pi1","pi2","pi3","X","Y","theta"]
a_outputs = ["pi1","pi2","pi3"]

# Buckingham based non-dimensionnalized models for RACECAR
a_racecar_M, a_racecar_X, a_racecar_y, a_racecar_c = createModels("racecar.xlsx",a_drop, a_outputs)
# Buckingham based non-dimensionnalized models for LIMO
a_limo_M, a_limo_X, a_limo_y, a_limo_c = createModels("limo.xlsx",a_drop, a_outputs)
# Buckingham based non-dimensionnalized models for XMAXX
a_Xmaxx_M, a_Xmaxx_X, a_Xmaxx_y, a_Xmaxx_c = createModels("Xmaxx.xlsx",a_drop, a_outputs)
# Buckingham based non-dimensionnalized models for merged files of all 3 small vehicles
a_merged_M, a_merged_X, a_merged_y, a_merged_c = createModels("merged.xlsx",a_drop, a_outputs)
# Buckingham based non-dimensionnalized models for CIVIC
a_civic_M, a_civic_X, a_civic_y, a_civic_c = createModels("civic.xlsx",a_drop, a_outputs)

#Packaging
a_models     =  [a_racecar_M, a_limo_M, a_Xmaxx_M, a_merged_M, a_civic_M]
a_X_tests     = [a_racecar_X, a_limo_X, a_Xmaxx_X, a_merged_X, a_civic_X]
a_y_tests     = [a_racecar_y, a_limo_y, a_Xmaxx_y, a_merged_y, a_civic_y]
a_corrections = [a_racecar_c, a_limo_c, a_Xmaxx_c, a_merged_c, a_civic_c]

# Test and print self, cross and shared predictions MAE (Mean Absolute Error)
var_holder = {}
var_holder, MAE_list = testModels(a_models, a_X_tests, a_y_tests, var_holder, a_corrections)
locals().update(var_holder)
#printTable()
printReport(MAE_list)

                                                              















