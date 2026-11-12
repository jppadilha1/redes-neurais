"""Funcoes comuns da rede de Kohonen (mapa auto-organizavel 4 x 4, 3 entradas).
Usado por todos os scripts. Nao precisa rodar este arquivo.
"""
import os
import numpy as np
import pandas as pd

ARQ_TREINO = "treinamento.csv"
ARQ_TESTE = "teste.csv"
PASTA = "saidas"
os.makedirs(PASTA, exist_ok=True)

ENTRADAS = ["x1", "x2", "x3"]
LINHAS_GRID, COLUNAS_GRID = 4, 4
N1 = LINHAS_GRID * COLUNAS_GRID          # 16 neuronios
ETA = 0.001                               # taxa de aprendizado
RAIO = 1                                  # raio de vizinhanca no grid
EPS = 1e-6                                # para quando nenhum peso muda mais que isso numa epoca
MAX_EPOCAS = 50_000

# classes definidas no enunciado pela numeracao das amostras de treinamento
CLASSES = {"A": (1, 20), "B": (21, 60), "C": (61, 120)}
CORES = {"A": "#4c9be8", "B": "#f5a142", "C": "#5cb85c", "-": "#e0e0e0"}

# posicao (linha, coluna) de cada neuronio no grid: neuronio 1 = (0,0), 2 = (0,1), ..., 16 = (3,3)
POS = np.array([(k // COLUNAS_GRID, k % COLUNAS_GRID) for k in range(N1)])
# vizinhos de cada neuronio: distancia no grid (horizontal + vertical) <= RAIO, como no diagrama
VIZINHOS = [np.where((np.abs(POS - POS[j]).sum(axis=1) <= RAIO) & (np.arange(N1) != j))[0] for j in range(N1)]


def ler(caminho):
    """Le o CSV e devolve DataFrame com 'amostra', x1, x2, x3.
    Aceita separador , ou ; e decimal com ponto ou virgula.
    Se as colunas nao se chamarem x1/x2/x3, usa as 3 ultimas colunas do arquivo."""
    df = pd.read_csv(caminho, sep=None, engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
    if not set(ENTRADAS).issubset(df.columns):
        df = df.rename(columns=dict(zip(df.columns[-3:], ENTRADAS)))
    df = df.dropna(subset=ENTRADAS).reset_index(drop=True)
    if "amostra" not in df.columns:
        df["amostra"] = np.arange(1, len(df) + 1)
    df["amostra"] = df["amostra"].astype(int)
    return df[["amostra"] + ENTRADAS]


def classe_da_amostra(numero):
    for c, (ini, fim) in CLASSES.items():
        if ini <= numero <= fim:
            return c
    return "-"


def vencedor(x, W):
    """Neuronio vencedor = menor distancia euclidiana entre x e os pesos (indice 0..15)."""
    return int(np.argmin(((W - x) ** 2).sum(axis=1)))


def carregar_pesos():
    p = pd.read_csv(os.path.join(PASTA, "parte1_pesos.csv"))
    return p[["w1", "w2", "w3"]].values


def carregar_neuronios():
    return pd.read_csv(os.path.join(PASTA, "parte1_neuronios.csv"))


def desenhar_grid(ax, rotulos, textos=None, titulo=""):
    """Desenha o grid 4 x 4 como no enunciado (circulos numerados ligados aos vizinhos)."""
    import matplotlib.pyplot as plt
    for j in range(N1):
        for k in VIZINHOS[j]:
            if k > j:
                (l1, c1), (l2, c2) = POS[j], POS[k]
                ax.plot([c1, c2], [l1, l2], color="gray", lw=1.5, zorder=1)
    for j in range(N1):
        l, c = POS[j]
        ax.add_patch(plt.Circle(
            (c, l), 0.32, color=CORES.get(rotulos[j], "#e0e0e0"), ec="black", zorder=2))
        ax.text(c, l - 0.07, str(j + 1), ha="center", va="center", fontsize=12, fontweight="bold", zorder=3)
        if textos:
            ax.text(c, l + 0.15, textos[j], ha="center", va="center", fontsize=7, zorder=3)
    ax.set(xlim=(-0.6, COLUNAS_GRID - 0.4), ylim=(LINHAS_GRID - 0.4, -0.6), aspect="equal", title=titulo)
    ax.axis("off")
