"""Funcoes comuns: leitura dos CSVs e camada escondida da RBF.
Usado por todos os scripts (parte1 ... parte5). Nao precisa rodar este arquivo.
"""
import os
import numpy as np
import pandas as pd

ARQ_TREINO = "treinamento.csv"
ARQ_TESTE = "teste.csv"
PASTA = "saidas"
os.makedirs(PASTA, exist_ok=True)


def ler(caminho):
    """Le o CSV e devolve um DataFrame com as colunas x1, x2, d.
    Aceita separador , ou ; e decimal com ponto ou virgula.
    Se as colunas nao se chamarem x1/x2/d, usa as 3 ultimas colunas do arquivo."""
    df = pd.read_csv(caminho, sep=None, engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
    if not {"x1", "x2", "d"}.issubset(df.columns):
        df = df.iloc[:, -3:]
        df.columns = ["x1", "x2", "d"]
    df = df[["x1", "x2", "d"]].dropna().reset_index(drop=True)
    df["d"] = df["d"].astype(int)
    return df


def gaussianas(X, centros, variancias):
    """Saida da camada escondida: phi_k(x) = exp(-||x - c_k||^2 / (2 sigma_k^2))."""
    dist2 = ((X[:, None, :] - centros[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-dist2 / (2 * variancias[None, :]))


def entrada_saida(X, centros, variancias):
    """Vetor que entra no neuronio de saida: [-1, phi_1, phi_2] (limiar com entrada fixa -1)."""
    phi = gaussianas(X, centros, variancias)
    return np.c_[-np.ones(len(X)), phi]


def carregar_centros():
    c = pd.read_csv(os.path.join(PASTA, "parte1_centros.csv"))
    return c[["c_x1", "c_x2"]].values, c["variancia"].values


def carregar_pesos():
    return pd.read_csv(os.path.join(PASTA, "parte2_pesos.csv"))["valor"].values
