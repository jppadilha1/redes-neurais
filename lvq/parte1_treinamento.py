"""PARTE 1 - Treinamento da LVQ-1 (um neuronio/prototipo por classe, taxa = 0.05).
Pesos iniciais: a primeira amostra de cada classe.
A cada amostra x, o neuronio vencedor j (menor distancia euclidiana) e ajustado:
    classe do vencedor == classe de x  ->  w_j <- w_j + eta * (x - w_j)   (aproxima)
    classe do vencedor != classe de x  ->  w_j <- w_j - eta * (x - w_j)   (afasta)
Para quando nenhum peso muda mais que EPS ao longo de uma epoca.
Gera:
  saidas/parte1_prototipos_iniciais.csv
  saidas/parte1_prototipos.csv          pesos finais (vetor representativo de cada classe)
  saidas/parte1_variacao_por_epoca.csv  maior mudanca de peso e erros de classificacao em cada epoca
  saidas/parte1_treino.csv              neuronio vencedor e classe atribuida a cada amostra de treino
  saidas/parte1_perfis.png              curvas das amostras de treino e prototipo de cada classe
  saidas/parte1_convergencia.png        variacao dos pesos por epoca
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dados import ler, vencedor, ARQ_TREINO, PASTA, ETA, EPS, MAX_EPOCAS, HORAS, CORES

treino, X = ler(ARQ_TREINO)
d = treino["classe"].astype(int).values
classes = sorted(int(c) for c in set(d))
print(f"Amostras de treino: {len(X)} | classes: {classes}")

# ---- pesos iniciais: primeira amostra de cada classe ----
idx_ini = [int(np.where(d == c)[0][0]) for c in classes]
W = X[idx_ini].copy()
cls_W = np.array(classes)
ini = pd.DataFrame(W, columns=HORAS)
ini.insert(0, "classe", cls_W)
ini.insert(0, "neuronio", range(1, len(classes) + 1))
ini["amostra_usada"] = treino["id"].values[idx_ini]
ini.to_csv(os.path.join(PASTA, "parte1_prototipos_iniciais.csv"), index=False)

# ---- treinamento ----
hist = []
for epoca in range(1, MAX_EPOCAS + 1):
    W_ant, erros = W.copy(), 0
    for x, dk in zip(X, d):
        j = vencedor(x, W)
        if cls_W[j] == dk:
            W[j] += ETA * (x - W[j])
        else:
            W[j] -= ETA * (x - W[j])
            erros += 1
    hist.append({"epoca": epoca, "maior_variacao_peso": np.abs(W - W_ant).max(), "erros_na_epoca": erros})
    if hist[-1]["maior_variacao_peso"] < EPS:
        break
hist = pd.DataFrame(hist)
hist.to_csv(os.path.join(PASTA, "parte1_variacao_por_epoca.csv"), index=False)
print(f"Convergiu em {epoca} epocas")

proto = pd.DataFrame(W, columns=HORAS)
proto.insert(0, "classe", cls_W)
proto.insert(0, "neuronio", range(1, len(classes) + 1))
proto.to_csv(os.path.join(PASTA, "parte1_prototipos.csv"), index=False)

# ---- conferencia no proprio treino ----
res = treino.copy()
res["neuronio_vencedor"] = [vencedor(x, W) + 1 for x in X]
res["classe_rede"] = cls_W[res["neuronio_vencedor"] - 1]
res["acertou"] = res["classe_rede"] == res["classe"].astype(int)
res.to_csv(os.path.join(PASTA, "parte1_treino.csv"), index=False)

pd.set_option("display.width", 200)
print("\nPrototipos iniciais:")
print(ini.round(4).to_string(index=False))
print("\nPrototipos finais:")
print(proto.round(4).to_string(index=False))
print(f"\nAcerto no treino: {res['acertou'].sum()} de {len(res)}")
print("\nMedia de cada classe (para comparar com o prototipo):")
print(treino.groupby("classe")[HORAS].mean().round(4).to_string())

# ---- figuras ----
fig, axs = plt.subplots(1, len(classes), figsize=(4 * len(classes), 3.8), sharey=True)
for ax, c, w in zip(axs, cls_W, W):
    for x in X[d == c]:
        ax.plot(HORAS, x, color=CORES.get(c, "gray"), alpha=.35, lw=1)
    ax.plot(HORAS, w, color="black", lw=2.5, marker="o", label="Protótipo (pesos)")
    ax.set(title=f"Classe {c}", xlabel="Hora")
    ax.grid(alpha=.3)
axs[0].set_ylabel("Potência")
axs[0].legend(fontsize=8)
fig.suptitle("Amostras de treino (cor) e protótipo aprendido (preto)")
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "parte1_perfis.png"), dpi=150)

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.semilogy(hist["epoca"], hist["maior_variacao_peso"])
ax.axhline(EPS, color="gray", ls="--", lw=1)
ax.set(xlabel="Época", ylabel="Maior variação de peso (log)", title=f"Convergência – {epoca} épocas")
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "parte1_convergencia.png"), dpi=150)
print(f"\nArquivos salvos em ./{PASTA}/")
