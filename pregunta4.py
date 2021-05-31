# -*- coding: utf-8 -*-
"""
Created on Sun May 29 23:45:19 2021

@author: Hp
"""

import numpy as np
import pandas as pd
dataset=pd.read_csv("audit_risk.csv")
#dataset=pd.read_csv("trial.csv")
print( dataset.head())
print("columnas del dataset ")
print(dataset.columns)
print("caracteristicas de cada columna")
print( dataset.dtypes)
#preprocesamiento
print("PREPROCESAMEINTO")
print("verificando vacios en los datos")
print(dataset.info(verbose=True,null_counts=True))
print("remplazando vacios por 0")
dataset=dataset.replace(np.nan,"0")
print(dataset.info(verbose=True,null_counts=True))
#creando red neuronal
X=dataset.loc[:,['TOTAL','numbers','Money_Value','PROB','CONTROL_RISK']]
Y=dataset["LOCATION_ID"]
print("valores unicos de la tablas")
print(Y.unique())
#cambiar de texto a numero
Y=Y.replace({"LOHARU":'0',"NUH":'0',"SAFIDON":'0'})
print("valores unicos de la tablas")
print(Y.unique())
#normalizando
#definimos modelo de entrenamiento
#X_test con 80 % de entrenamiento
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.20)
#escalando la funciones de entrenamiento
from sklearn.preprocessing import StandardScaler
escalaX=StandardScaler()
X_train=escalaX.fit_transform(X_train)
X_test=escalaX.transform(X_test)
#entrenamiento y predicciones de la red neuronal
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000)
mlp.fit(X_train,y_train)
prediccion=mlp.predict(X_test)
#evaluacion del algoritmo
from sklearn.metrics import confusion_matrix
print("matriz de confusion")
print(confusion_matrix(y_test,prediccion))
from sklearn.metrics import classification_report
print("reporte de clasificacion")
print(classification_report(y_test,prediccion))
