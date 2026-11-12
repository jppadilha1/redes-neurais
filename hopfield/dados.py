"""Funcoes comuns da rede de Hopfield (45 neuronios, imagens 9 x 5).
Usado por todos os scripts. Nao precisa rodar este arquivo.

Formato dos CSVs: uma imagem por linha, 45 colunas p01..p45 com -1 (branco) ou +1 (escuro),
lidas linha a linha da imagem (p01..p05 = 1a linha, p06..p10 = 2a linha, ...).
Colunas extras aceitas: 'padrao' (qual imagem original) e 'situacao'.
"""
import os
import numpy as np
import pandas as pd

ARQ_TREINO = "treinamento.csv"
ARQ_TESTE = "teste.csv"
PASTA = "saidas"
os.makedirs(PASTA, exist_ok=True)

LINHAS, COLUNAS = 9, 5
N = LINHAS * COLUNAS          # 45 neuronios
BETA = 1000                   # tangente hiperbolica com beta muito grande (~ funcao sinal)
SEMENTE = 2026


def ler(caminho):
    """Devolve (metadados, matriz de imagens k x 45 com valores -1/+1)."""
    df = pd.read_csv(caminho, sep=None, engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]
    pix = [c for c in df.columns if c.startswith("p") and c[1:].isdigit()]
    if len(pix) != N:                              # sem nomes p01..p45: usa as 45 ultimas colunas
        pix = list(df.columns[-N:])
    X = df[pix].apply(pd.to_numeric).values.astype(int)
    X = np.where(X > 0, 1, -1)                     # aceita 0/1 ou -1/+1
    meta = df.drop(columns=pix)
    if "padrao" in meta:
        meta["padrao"] = meta["padrao"].astype(str)
    return meta.reset_index(drop=True), X


def treinar(Z):
    """Regra do produto externo: W = (1/n) * sum(z z^T) - (p/n) * I  (diagonal zerada)."""
    p, n = Z.shape
    return (Z.T @ Z) / n - (p / n) * np.eye(n)


def g(u):
    """Funcao de ativacao: tangente hiperbolica com beta muito grande."""
    return np.tanh(BETA * u)


def energia(v, W):
    return -0.5 * v @ W @ v


def recuperar(v0, W, rng, max_iter=100):
    """Dinamica da rede: atualiza os neuronios um a um (ordem aleatoria) ate nenhum mudar.
    Se u = 0, o neuronio mantem o valor anterior.
    Devolve (estado final, numero de iteracoes, energia a cada iteracao)."""
    v = v0.astype(float).copy()
    energias = [energia(v, W)]
    for it in range(1, max_iter + 1):
        anterior = v.copy()
        for i in rng.permutation(len(v)):
            u = W[i] @ v
            if u != 0:
                v[i] = np.sign(g(u))
        energias.append(energia(v, W))
        if np.array_equal(v, anterior):
            return v.astype(int), it, energias
    return v.astype(int), max_iter, energias


def identificar(v, Z, nomes):
    """Diz em que estado a rede parou: um padrao armazenado, o inverso de um padrao ou estado espurio."""
    for z, nome in zip(Z, nomes):
        if np.array_equal(v, z):
            return f"Padrão {nome}"
    for z, nome in zip(Z, nomes):
        if np.array_equal(v, -z):
            return f"Inverso do padrão {nome}"
    return "Estado espúrio"


def desenhar(ax, v, titulo=""):
    """Desenha a imagem 9 x 5 (cinza = +1, branco = -1)."""
    from matplotlib.colors import ListedColormap
    img = (np.asarray(v).reshape(LINHAS, COLUNAS) > 0).astype(int)
    ax.imshow(img, cmap=ListedColormap(["white", "#9e9e9e"]), vmin=0, vmax=1)
    ax.set_xticks(np.arange(-.5, COLUNAS, 1), minor=True)
    ax.set_yticks(np.arange(-.5, LINHAS, 1), minor=True)
    ax.grid(which="minor", color="black", lw=0.8)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    ax.set_title(titulo, fontsize=9)
