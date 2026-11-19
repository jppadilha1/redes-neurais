"""Funcoes comuns da rede LVQ-1 (6 entradas = potencia das 7h as 12h, 4 classes).
Usado por todos os scripts. Nao precisa rodar este arquivo.
"""
import os
import numpy as np
import pandas as pd

ARQ_TREINO = "treinamento.csv"
ARQ_TESTE = "teste.csv"
PASTA = "saidas"
os.makedirs(PASTA, exist_ok=True)

ETA = 0.05            # taxa de aprendizagem (gama do enunciado)
EPS = 1e-6            # para quando nenhum peso muda mais que isso em uma epoca
MAX_EPOCAS = 10_000
HORAS = ["7h", "8h", "9h", "10h", "11h", "12h"]
CORES = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}


def ler(caminho):
    """Le o CSV e devolve (DataFrame com id + 6 entradas + classe, matriz X).
    Aceita separador , ou ; e decimal com ponto ou virgula.
    Colunas reconhecidas: 'amostra' ou 'dia' (id), 'classe' (opcional no teste);
    as demais colunas numericas sao as 6 entradas, na ordem do arquivo."""
    df = pd.read_csv(caminho, sep=None, engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
    col_id = next((c for c in df.columns if c in ("amostra", "dia")), None)
    col_cl = next((c for c in df.columns if c.startswith("classe")), None)
    entradas = [c for c in df.columns if c not in (col_id, col_cl)]
    if len(entradas) != 6:
        raise ValueError(f"{caminho}: esperava 6 colunas de potencia, encontrei {len(entradas)}: {entradas}")
    out = df[entradas].copy()
    out.columns = HORAS
    out.insert(0, "id", df[col_id].astype(int) if col_id else np.arange(1, len(df) + 1))
    out["classe"] = df[col_cl] if col_cl and df[col_cl].notna().any() else np.nan
    out = out.dropna(subset=HORAS).reset_index(drop=True)
    return out, out[HORAS].values.astype(float)


def vencedor(x, W):
    """Neuronio (prototipo) com menor distancia euclidiana ate x."""
    return int(np.argmin(((W - x) ** 2).sum(axis=1)))


def carregar_prototipos():
    p = pd.read_csv(os.path.join(PASTA, "parte1_prototipos.csv"))
    return p[HORAS].values, p["classe"].values.astype(int)
