import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle

##########################################################
# Código de Entrenamiento - Modelo de Price Prediction
##########################################################

# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename)).set_index('id')
    X_train = df.drop(['Exited'],axis=1)
    y_train = df[['Exited']]
    # Balancemos los datos de Train
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(filename, ' cargado correctamente')
    # Entrenamos el modelo con toda la muestra
    lgb = LGBMClassifier(random_state=42)
    lgb.fit(X_train, y_train)
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    filename = '../models/best_model.pkl'
    pickle.dump(lgb, open(filename, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento completo
def main():
    read_file_csv('churn_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()
