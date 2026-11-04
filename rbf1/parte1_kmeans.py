"""PARTE 1 - Treinamento da camada escondida com k-means.
Usa so os padroes com presenca de radiacao (d = +1), k = 2 clusters.
Gera:
  saidas/parte1_centros.csv          centro e variancia de cada cluster
  saidas/parte1_clusters_treino.csv  cada amostra d=+1 com o cluster a que pertence
  saidas/parte1_clusters.png         grafico dos clusters
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from dados import ler, ARQ_TREINO, PASTA

K = 2

treino = ler(ARQ_TREINO)
positivos = treino[treino["d"] == 1].copy()
P = positivos[["x1", "x2"]].values
print(f"Amostras de treino: {len(treino)} | com radiacao (d=+1): {len(P)}")

# ---- k-means (scikit-learn); n_init=10 roda 10 inicializacoes e fica com a melhor ----
km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(P)
centros, rotulos = km.cluster_centers_, km.labels_

# numera os clusters em ordem crescente de x1 (so para o resultado ficar sempre igual)
ordem = np.argsort(centros[:, 0])
centros = centros[ordem]
rotulos = np.argsort(ordem)[rotulos]

# ---- variancia: media das distancias ao quadrado de cada ponto ate o seu centro ----
variancias = np.array([((P[rotulos == k] - centros[k]) ** 2).sum(axis=1).mean() for k in range(K)])

tabela = pd.DataFrame({
    "cluster": np.arange(1, K + 1),
    "c_x1": centros[:, 0], "c_x2": centros[:, 1],
    "variancia": variancias,
    "desvio_sigma": np.sqrt(variancias),
    "n_amostras": np.bincount(rotulos, minlength=K),
})
tabela.to_csv(os.path.join(PASTA, "parte1_centros.csv"), index=False)

positivos["cluster"] = rotulos + 1
positivos.to_csv(os.path.join(PASTA, "parte1_clusters_treino.csv"), index_label="linha_original")

print("\nResultado do k-means:")
print(tabela.round(4).to_string(index=False))
print(f"Iteracoes ate convergir: {km.n_iter_}")

# ---- grafico ----
fig, ax = plt.subplots(figsize=(8.5, 6))
neg = treino[treino["d"] == -1]
ax.scatter(neg["x1"], neg["x2"], marker="x", c="gray", label="Ausência (d = -1)")
cores = ["tab:blue", "tab:orange"]
for k in range(K):
    pts = P[rotulos == k]
    ax.scatter(pts[:, 0], pts[:, 1], c=cores[k], label=f"Presença - cluster {k+1}")
    ax.scatter(*centros[k], marker="*", s=350, c=cores[k], edgecolors="black", zorder=5)
    ax.add_patch(plt.Circle(centros[k], np.sqrt(variancias[k]), fill=False, ls="--", color=cores[k]))
ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x1", ylabel="x2", aspect="equal",
       title="k-means nos padrões com radiação\n(★ centro, tracejado = raio σ)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "parte1_clusters.png"), dpi=150)
print(f"\nArquivos salvos em ./{PASTA}/")
