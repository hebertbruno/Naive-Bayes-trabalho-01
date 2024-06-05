from naive_bayes import NaiveBayes

def train_model(X_treino, y_treino):
    model = NaiveBayes()
    model.fit(X_treino, y_treino)
    return model
