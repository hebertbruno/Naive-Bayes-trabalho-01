from metricas import calc_acuracia, coeficiente_pearson, erro_medio_quadratico

def test_model(model, X_teste, y_teste):
    y_prev = model.previsao(X_teste)
    
    acuracia = calc_acuracia(y_teste, y_prev)
    
    class_mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    y_test_numeric = [class_mapping[cls] for cls in y_teste]
    y_prev_numeric = [class_mapping[cls] for cls in y_prev]
    
    pearson = coeficiente_pearson(y_test_numeric, y_prev_numeric)
    mse = erro_medio_quadratico(y_test_numeric, y_prev_numeric)
    
    return acuracia, pearson, mse
