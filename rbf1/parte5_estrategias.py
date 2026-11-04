"""PARTE 5 - Experimentos para embasar as estrategias de melhoria.
Nao altera os resultados das partes 1-4; so testa alternativas e compara:
  A) mais neuronios na camada escondida (k = 2..8)
  B) gaussianas mais largas (variancia multiplicada por um fator)
  C) centros tambem para a classe -1 (k-means em cada classe)
  D) perceptron simples (uma reta), para referencia
Gera:
  saidas/parte5_estrategias.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Perceptron
from dados import ler, entrada_saida, ARQ_TREINO, ARQ_TESTE, PASTA

treino, teste = ler(ARQ_TREINO), ler(ARQ_TESTE)
X, d = treino[["x1", "x2"]].values, treino["d"].values
Xt, dt = teste[["x1", "x2"]].values, teste["d"].values


def kmeans_var(P, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(P)
    c, r = km.cluster_centers_, km.labels_
    v = np.array([max(((P[r == i] - c[i]) ** 2).sum(1).mean(), 1e-4) for i in range(k)])
    return c, v


def regra_delta(Z, d, eta=0.01, eps=1e-7, semente=0):
    W = np.random.default_rng(semente).uniform(0, 1, Z.shape[1])
    ant = 0.5 * np.mean((d - Z @ W) ** 2)
    for _ in range(100_000):
        for z, dk in zip(Z, d):
            W += eta * (dk - z @ W) * z
        atual = 0.5 * np.mean((d - Z @ W) ** 2)
        if abs(atual - ant) <= eps:
            break
        ant = atual
    return W


def avalia(centros, variancias):
    W = regra_delta(entrada_saida(X, centros, variancias), d)
    acc = lambda A, b: (np.where(entrada_saida(A, centros, variancias) @ W >= 0, 1, -1) == b).mean() * 100
    return acc(X, d), acc(Xt, dt)


linhas = []
P = X[d == 1]
for k in range(2, 9):                                           # A
    tr, te = avalia(*kmeans_var(P, k))
    linhas.append(("A) mais neuronios (so d=+1)", f"k={k}", tr, te))

c2, v2 = kmeans_var(P, 2)
for f in [0.5, 1, 2, 3, 4]:                                     # B
    tr, te = avalia(c2, v2 * f)
    linhas.append(("B) largura das gaussianas (k=2)", f"variancia x{f}", tr, te))

for k in [1, 2, 3]:                                             # C
    cp, vp = kmeans_var(P, k)
    cn, vn = kmeans_var(X[d == -1], k)
    tr, te = avalia(np.r_[cp, cn], np.r_[vp, vn])
    linhas.append(("C) centros nas duas classes", f"{k}+{k} neuronios", tr, te))

pc = Perceptron(max_iter=10_000, random_state=0).fit(X, d)       # D
linhas.append(("D) perceptron simples (reta)", "-", pc.score(X, d) * 100, pc.score(Xt, dt) * 100))

tab = pd.DataFrame(linhas, columns=["estrategia", "configuracao", "acerto_treino_%", "acerto_teste_%"])
tab.to_csv(os.path.join(PASTA, "parte5_estrategias.csv"), index=False)
print(tab.round(1).to_string(index=False))
print(f"\nArquivo salvo em ./{PASTA}/parte5_estrategias.csv")
