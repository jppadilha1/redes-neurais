"""PARTE 1 - Armazenamento dos 4 padroes (regra do produto externo).
Gera:
  saidas/parte1_matriz_W.csv       matriz de pesos 45 x 45
  saidas/parte1_padroes.png        as 4 imagens armazenadas
  saidas/parte1_estabilidade.csv   confere se cada padrao e um ponto de equilibrio da rede
  saidas/parte1_similaridade.csv   quantos pixels cada par de padroes tem em comum
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dados import ler, treinar, recuperar, desenhar, energia, ARQ_TREINO, PASTA, SEMENTE, N

meta, Z = ler(ARQ_TREINO)
nomes = meta["padrao"].tolist() if "padrao" in meta else [str(i + 1) for i in range(len(Z))]
W = treinar(Z)
print(f"Padroes armazenados: {len(Z)} | neuronios: {N} | capacidade teorica ~ 0,138 x {N} = {0.138*N:.1f}")

pd.DataFrame(W).to_csv(os.path.join(PASTA, "parte1_matriz_W.csv"), index=False, float_format="%.6f")

rng = np.random.default_rng(SEMENTE)
est = []
for z, nome in zip(Z, nomes):
    v, it, _ = recuperar(z, W, rng)
    est.append({"padrao": nome, "estavel": bool(np.array_equal(v, z)), "energia": energia(z, W)})
est = pd.DataFrame(est)
est.to_csv(os.path.join(PASTA, "parte1_estabilidade.csv"), index=False)
print("\nCada padrao apresentado sem ruido volta para ele mesmo?")
print(est.round(4).to_string(index=False))

iguais = (Z @ Z.T + N) // 2
sim = pd.DataFrame(iguais, index=[f"padrao {n}" for n in nomes], columns=[f"padrao {n}" for n in nomes])
sim.to_csv(os.path.join(PASTA, "parte1_similaridade.csv"))
print("\nPixels iguais entre os padroes (de 45):")
print(sim.to_string())

fig, axs = plt.subplots(1, len(Z), figsize=(2 * len(Z), 3.2))
for ax, z, nome in zip(axs, Z, nomes):
    desenhar(ax, z, f"Padrão {nome}")
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "parte1_padroes.png"), dpi=150)
print(f"\nArquivos salvos em ./{PASTA}/")
