import pandas as pd
from sklearn.model_selection import StratifiedKFold

def load_data(filename):
    data = pd.read_excel(filename)
    X = data.iloc[:, :-1].values.tolist()  # Todas as colunas, exceto a última, caracteristicas
    y = data.iloc[:, -1].values.tolist()   # Apenas a última coluna, Classes
    return X, y

def create_folds(X, y, num_folds=5):
    skf = StratifiedKFold(n_splits=num_folds)
    folds = []
    for treino_index, teste_index in skf.split(X, y):
        X_treino = [X[i] for i in treino_index] # Lista das caracteristicas para treinamento
        y_treino = [y[i] for i in treino_index] # Lista das classes para treinamento
        X_teste = [X[i] for i in teste_index] #Lista das caracteristicas para teste
        y_teste = [y[i] for i in teste_index] # Lista das classes para teste
        folds.append((X_treino, y_treino, X_teste, y_teste))
    return folds
