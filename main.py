import numpy as np
from data import load_data, create_folds
from treino import train_model
from testes import test_model
from metricas import media

# Carregar os dados e criar as pastas
filename = 'iris_dataset.xlsx'
X, y = load_data(filename)
folds = create_folds(X, y)

acuracias = []
coeficientes_pearson = []
mse_values = []

for index, fold in enumerate(folds, start=1):
    X_treino, y_treino, X_teste, y_teste = fold
    
    model = train_model(X_treino, y_treino)
    acuracia, pearson, mse = test_model(model, X_teste, y_teste)
    
    
    acuracias.append(acuracia)
    coeficientes_pearson.append(pearson)
    mse_values.append(mse)
    

media_acuracia = media(acuracias)
media_pearson = media(coeficientes_pearson)
media_mse = media(mse_values)

print("Acurácias:", acuracias)
print("Acurácia Média:", media_acuracia)
print("Coeficientes de Pearson:", coeficientes_pearson)
print("Coeficiente de Pearson Médio:", media_pearson)
print("Valores do Erro Médio Quadrático:", mse_values)
print("MSE Médio:", media_mse)


