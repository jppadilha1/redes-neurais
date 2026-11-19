"""PARTE 2 - Classificacao dos 8 dias de teste com a LVQ-1 treinada.
A classe de cada dia e a classe do prototipo mais proximo (distancia euclidiana).
Gera:
  saidas/parte2_classificacao.csv   dia, 6 potencias, distancia a cada prototipo, classe atribuida
  saidas/parte2_perfis_teste.png    curva de cada dia de teste junto ao prototipo da classe atribuida
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dados import ler, carregar_prototipos, ARQ_TESTE, PASTA, HORAS, CORES

W, cls_W = carregar_prototipos()
teste, X = ler(ARQ_TESTE)

dist = np.sqrt(((X[:, None, :] - W[None, :, :]) ** 2).sum(axis=2))
j = dist.argmin(axis=1)
res = teste[["id"] + HORAS].rename(columns={"id": "dia"}).copy()
for k, c in enumerate(cls_W):
    res[f"dist_classe_{c}"] = dist[:, k]
res["neuronio_vencedor"] = j + 1
res["classe"] = cls_W[j]
ordenadas = np.sort(dist, axis=1)
res["margem"] = ordenadas[:, 1] - ordenadas[:, 0]     # quanto o 2o prototipo esta mais longe que o 1o
res.to_csv(os.path.join(PASTA, "parte2_classificacao.csv"), index=False)

pd.set_option("display.width", 220)
print(res.round(4).to_string(index=False))

n = len(res)
cols = 4
fig, axs = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(4 * cols, 3.3 * np.ceil(n / cols)),
                        sharey=True, squeeze=False)
for ax, (_, r), x, k in zip(axs.flat, res.iterrows(), X, j):
    c = cls_W[k]
    ax.plot(HORAS, W[k], color=CORES.get(c, "gray"), lw=2.5, marker="o", label=f"Protótipo classe {c}")
    ax.plot(HORAS, x, color="black", lw=1.5, ls="--", marker="s", ms=4, label=f"Dia {int(r['dia'])}")
    ax.set_title(f"Dia {int(r['dia'])} → classe {c}")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
for ax in axs.flat[n:]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "parte2_perfis_teste.png"), dpi=150)
print(f"\nArquivos salvos em ./{PASTA}/")
