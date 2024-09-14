#####################################################
# Script de Preparación de Datos-Train, Test, Score
#####################################################

import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/', filename)).set_index('id')
    print(filename, ' cargado correctamente')
    return df

# Realizamos la transformación de datos
def data_preparation(df):
    # Codificamos los valores categoricos
    label_encoder = LabelEncoder()
    df['Geography'] =  label_encoder.fit_transform(df['Geography'])
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    # Eliminamos variable no relevante
    df = df.drop('Surname',axis=1)
    df.drop(['CustomerId'], axis=1, inplace=True)
    # Transformamos para reducir el sesgo por los atípicos
    columns=['CreditScore','Balance','EstimatedSalary','Age','NumOfProducts']
    for col in columns:
        df[col]=winsorize(df[col],limits=[0.05,0.1],inclusive=(True,True))

    print('Transformación de datos completa')
    return df

# Exportamos la matriz de datos con las columnas seleccionadas
# def data_exporting(df, features, filename):
def data_exporting_train_test(df, filename_train,filename_test):
    # Sepramos la data en Train y Test
    X = df.drop(['Exited'],axis=1)
    y = df[['Exited']]
    # Dataset de Train (80%) y Test (20%)
    train=pd.concat([X_train,y_train],axis=1)
    test=pd.concat([X_test,y_test],axis=1)
    train.to_csv(os.path.join('../data/processed/', filename_train))
    test.to_csv(os.path.join('../data/processed/', filename_test))
    print(filename_train, '-' ,filename_test,' exportados correctamente en la carpeta processed')
