#metrica.py

def media(values):
    return sum(values) / len(values)

def desvio(values):
    media_value = media(values)
    variancia = sum((x - media_value) ** 2 for x in values) / len(values)
    return variancia ** 0.5

def calc_acuracia(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

def coeficiente_pearson(y_true, y_pred):
    n = len(y_true)
    sum_x = sum(y_true)
    sum_y = sum(y_pred)
    sum_xy = sum(x * y for x, y in zip(y_true, y_pred))
    sum_x2 = sum(x ** 2 for x in y_true)
    sum_y2 = sum(y ** 2 for y in y_pred)
    
    numerador = sum_xy - (sum_x * sum_y / n)
    denominador = ((sum_x2 - (sum_x ** 2 / n)) * (sum_y2 - (sum_y ** 2 / n))) ** 0.5
    if denominador == 0:
        return 0
    return numerador / denominador

def erro_medio_quadratico(y_true, y_prev):
    return sum((y_true[i] - y_prev[i]) ** 2 for i in range(len(y_true))) / len(y_true)
