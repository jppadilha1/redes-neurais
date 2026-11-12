"""PARTE 1 - Treinamento da rede de Kohonen e neuronios de cada classe (questao 1).
N1 = 16 (grid 4 x 4), eta = 0.001, raio de vizinhanca = 1.
Pesos iniciais = as 16 primeiras amostras de treinamento.
A cada amostra: o vencedor e ajustado com eta e os vizinhos (raio 1) com eta/2:
    w_j <- w_j + eta   * (x - w_j)        (vencedor)
    w_v <- w_v + eta/2 * (x - w_v)        (vizinhos do vencedor)
Para quando nenhum peso muda mais que EPS ao longo de uma epoca.
Gera:
  saidas/parte1_pesos.csv              pesos finais dos 16 neuronios
  saidas/parte1_variacao_por_epoca.csv maior mudanca de peso em cada epoca
  saidas/parte1_vencedores_treino.csv  neuronio vencedor de cada amostra de treino e sua classe
  saidas/parte1_neuronios.csv          quantas amostras de A, B e C cada neuronio venceu e a classe dele
  saidas/parte1_grid_classes.png       grid 4 x 4 colorido pela classe de cada neuronio
  saidas/parte1_mapa_3d.png            amostras e pesos dos neuronios no espaco (x1, x2, x3)
  saidas/parte1_convergencia.png       variacao dos pesos ao longo das epocas
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dados import (ler, classe_da_amostra, vencedor, desenhar_grid, ARQ_TREINO, PASTA, ENTRADAS,
                   N1, POS, VIZINHOS, ETA, EPS, MAX_EPOCAS, CLASSES, CORES)

treino = ler(ARQ_TREINO)
X = treino[ENTRADAS].values
treino["classe"] = treino["amostra"].map(classe_da_amostra)
print(f"Amostras de treino: {len(X)} | " + " | ".join(f"{c}: {(treino['classe']==c).sum()}" for c in CLASSES))

# ---- treinamento ----
W = X[:N1].copy()                       # pesos iniciais = 16 primeiras amostras
variacao = []
inicio = time.time()
for epoca in range(1, MAX_EPOCAS + 1):
    W_ant = W.copy()
    for x in X:
        j = vencedor(x, W)
        W[j] += ETA * (x - W[j])
        W[VIZINHOS[j]] += (ETA / 2) * (x - W[VIZINHOS[j]])
    variacao.append(np.abs(W - W_ant).max())
    if variacao[-1] < EPS:
        break
print(f"Convergiu em {epoca} epocas ({time.time() - inicio:.1f}s)")

pesos = pd.DataFrame({"neuronio": range(1, N1 + 1), "linha_grid": POS[:, 0] + 1, "coluna_grid": POS[:, 1] + 1,
                      "w1": W[:, 0], "w2": W[:, 1], "w3": W[:, 2]})
pesos.to_csv(os.path.join(PASTA, "parte1_pesos.csv"), index=False)
pd.DataFrame({"epoca": range(1, len(variacao) + 1), "maior_variacao_peso": variacao}).to_csv(
    os.path.join(PASTA, "parte1_variacao_por_epoca.csv"), index=False)

# ---- questao 1: quais neuronios respondem a cada classe ----
treino["neuronio_vencedor"] = [vencedor(x, W) + 1 for x in X]
treino.to_csv(os.path.join(PASTA, "parte1_vencedores_treino.csv"), index=False)

cont = pd.crosstab(treino["neuronio_vencedor"], treino["classe"]).reindex(
    index=range(1, N1 + 1), columns=list(CLASSES), fill_value=0)
neur = cont.reset_index().rename(columns={"neuronio_vencedor": "neuronio"})
neur["total"] = cont.sum(axis=1).values
neur["classe"] = [cont.loc[j].idxmax() if cont.loc[j].sum() > 0 else "-" for j in cont.index]
neur["pureza_%"] = [100 * cont.loc[j].max() / cont.loc[j].sum() if cont.loc[j].sum() > 0 else np.nan
                    for j in cont.index]
neur.to_csv(os.path.join(PASTA, "parte1_neuronios.csv"), index=False)

pd.set_option("display.width", 200)
print("\nAmostras vencidas por cada neuronio:")
print(neur.to_string(index=False))
print("\nNeuronios de cada classe:")
for c in CLASSES:
    print(f"  Classe {c}: {neur.loc[neur['classe'] == c, 'neuronio'].tolist()}")
print(f"  Nao venceram nenhuma amostra: {neur.loc[neur['classe'] == '-', 'neuronio'].tolist()}")

# ---- figuras ----
fig, ax = plt.subplots(figsize=(6, 6))
textos = [f"A{r.A} B{r.B} C{r.C}" if r.total else "nenhuma" for r in neur.itertuples()]
desenhar_grid(ax, neur["classe"].tolist(), textos, "Neurônios do grid por classe")
for c in list(CLASSES) + ["-"]:
    ax.scatter([], [], s=150, color=CORES[c], ec="black", label=f"Classe {c}" if c != "-" else "Sem amostras")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "parte1_grid_classes.png"), dpi=150)

fig = plt.figure(figsize=(7.5, 6))
ax = fig.add_subplot(projection="3d")
for c in CLASSES:
    p = treino[treino["classe"] == c]
    ax.scatter(p["x1"], p["x2"], p["x3"], s=12, color=CORES[c], alpha=.6, label=f"Amostras classe {c}")
for j in range(N1):
    for k in VIZINHOS[j]:
        if k > j:
            ax.plot(*zip(W[j], W[k]), color="black", lw=1)
ax.scatter(W[:, 0], W[:, 1], W[:, 2], s=60, color="red", ec="black", depthshade=False, label="Neurônios (pesos)")
for j in range(N1):
    ax.text(*W[j], f" {j+1}", fontsize=8)
ax.set(xlabel="x1", ylabel="x2", zlabel="x3", title="Amostras de treino e grid de Kohonen após o treinamento")
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "parte1_mapa_3d.png"), dpi=150)

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.semilogy(range(1, len(variacao) + 1), variacao)
ax.axhline(EPS, color="gray", ls="--", lw=1)
ax.set(xlabel="Época", ylabel="Maior variação de peso (log)", title=f"Convergência – {epoca} épocas")
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "parte1_convergencia.png"), dpi=150)
print(f"\nArquivos salvos em ./{PASTA}/")
