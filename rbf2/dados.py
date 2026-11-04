"""Funcoes comuns da RBF de aproximacao funcional (3 entradas, 1 saida).
Usado por todos os scripts. Nao precisa rodar este arquivo.
"""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ARQ_TREINO = "treinamento.csv"
ARQ_TESTE = "teste.csv"
PASTA = "saidas"
os.makedirs(PASTA, exist_ok=True)

ENTRADAS = ["x1", "x2", "x3"]
TOPOLOGIAS = {"Rede 1": 5, "Rede 2": 10, "Rede 3": 15}   # N1 = neuronios na camada escondida
SEMENTES = {"T1": 1, "T2": 2, "T3": 3}                     # uma semente diferente por treinamento
ETA, EPS, MAX_EPOCAS = 0.01, 1e-7, 200_000


def ler(caminho):
    """Le o CSV e devolve DataFrame com x1, x2, x3, d.
    Aceita separador , ou ; e decimal com ponto ou virgula.
    Se as colunas nao tiverem esses nomes, usa as 4 ultimas colunas do arquivo."""
    df = pd.read_csv(caminho, sep=None, engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
    if not set(ENTRADAS + ["d"]).issubset(df.columns):
        df = df.iloc[:, -4:]
        df.columns = ENTRADAS + ["d"]
    return df[ENTRADAS + ["d"]].dropna().reset_index(drop=True)


def kmeans(X, k):
    """Camada escondida: centros pelo k-means e variancia = media de ||x - c||^2 em cada cluster."""
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    c, r = km.cluster_centers_, km.labels_
    v = np.array([((X[r == i] - c[i]) ** 2).sum(axis=1).mean() for i in range(k)])
    v = np.where(v > 0, v, 1e-4)        # protege cluster com 1 amostra so
    return c, v, r


def entrada_saida(X, centros, variancias):
    """Vetor que entra no neuronio de saida: [-1, phi_1, ..., phi_N1]."""
    dist2 = ((X[:, None, :] - centros[None, :, :]) ** 2).sum(axis=2)
    phi = np.exp(-dist2 / (2 * variancias[None, :]))
    return np.c_[-np.ones(len(X)), phi]


def regra_delta(Z, d, semente):
    """Treina a camada de saida (ajuste amostra a amostra).
    Pesos iniciais aleatorios em [0, 1]; para quando |EQM_atual - EQM_anterior| <= EPS.
    Devolve pesos iniciais, pesos finais e historico do EQM (posicao 0 = antes de treinar)."""
    rng = np.random.default_rng(semente)
    W0 = rng.uniform(0, 1, Z.shape[1])
    W = W0.copy()
    eqm = lambda W: 0.5 * np.mean((d - Z @ W) ** 2)
    hist = [eqm(W)]
    for _ in range(MAX_EPOCAS):
        for z, dk in zip(Z, d):
            W += ETA * (dk - z @ W) * z
        hist.append(eqm(W))
        if abs(hist[-1] - hist[-2]) <= EPS:
            break
    return W0, W, np.array(hist)


def carregar_modelo(rede, treino):
    """Le centros, variancias e pesos salvos pela parte 1/2."""
    c = pd.read_csv(os.path.join(PASTA, "parte1_centros.csv"))
    c = c[c["rede"] == rede]
    p = pd.read_csv(os.path.join(PASTA, "parte1_pesos_finais.csv"))
    p = p[(p["rede"] == rede) & (p["treinamento"] == treino)]
    W = p.sort_values("indice")["valor"].values
    return c[ENTRADAS].values, c["variancia"].values, W
