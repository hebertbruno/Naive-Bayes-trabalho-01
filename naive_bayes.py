import math
from metricas import media, desvio

def probabilidade_gaussiana(x, media, desvio):
    expoente = -((x - media) ** 2) / (2 * desvio ** 2)
    probability = (math.exp(expoente) / (math.sqrt(2 * math.pi) * desvio))
    return probability

def calcular_estatistica_classe(X_c):
    medias = [media(column) for column in zip(*X_c)]
   
    desvios = [desvio(column) for column in zip(*X_c)]
   
    return medias, desvios

class NaiveBayes:
    def fit(self, X_treino, y_treino):
        self.classes = set(y_treino)
        self.priors = {}
        self.medias = {}
        self.desvios = {}
        
        for c in self.classes:
            X_c = [X_treino[i] for i in range(len(X_treino)) if y_treino[i] == c]
            self.priors[c] = len(X_c) / len(X_treino)
        
            self.medias[c], self.desvios[c] = calcular_estatistica_classe(X_c)

    def previsao(self, X_teste):
        previsoes = []
        for x in X_teste:
            posteriors = []
            for c in self.classes:
                prior = math.log(self.priors[c])
                condicional = sum(math.log(probabilidade_gaussiana(xi, self.medias[c][i], self.desvios[c][i])) for i, xi in enumerate(x)) # probabilidade condicional
                posterior = prior + condicional
               

                posteriors.append((posterior, c))
            previsoes.append(max(posteriors)[1])
        return previsoes
