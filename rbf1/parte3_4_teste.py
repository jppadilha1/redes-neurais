"""PARTES 3 e 4 - Pos-processamento (funcao sinal) e validacao com o conjunto de teste.
y_pos = 1 se y >= 0, senao -1 (aplicado so no teste).
Gera:
  saidas/parte4_resultado_teste.csv  amostra, x1, x2, d, y, y_pos, acertou
  saidas/parte4_resumo.csv           taxa de acerto no teste (e no treino, para comparar)
  saidas/parte4_fronteira.png        regiao classificada como presenca/ausencia + amostras
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dados import ler, entrada_saida, carregar_centros, carregar_pesos, ARQ_TREINO, ARQ_TESTE, PASTA


def sinal(y):  # parte 3: pos-processamento
    return np.where(y >= 0, 1, -1)


centros, variancias = carregar_centros()
W = carregar_pesos()
rede = lambda X: entrada_saida(X, centros, variancias) @ W

teste = ler(ARQ_TESTE)
treino = ler(ARQ_TREINO)

res = teste.copy()
res.insert(0, "amostra", np.arange(1, len(res) + 1))
res["y"] = rede(teste[["x1", "x2"]].values)
res["y_pos"] = sinal(res["y"])
res["acertou"] = res["y_pos"] == res["d"]
acc_teste = res["acertou"].mean() * 100
acc_treino = (sinal(rede(treino[["x1", "x2"]].values)) == treino["d"]).mean() * 100

print(res.round(4).to_string(index=False))
print(f"\nTaxa de acerto no TESTE : {acc_teste:.1f}%  ({res['acertou'].sum()}/{len(res)})")
print(f"Taxa de acerto no TREINO: {acc_treino:.1f}%  (so para comparacao)")

res.to_csv(os.path.join(PASTA, "parte4_resultado_teste.csv"), index=False)
pd.DataFrame({"conjunto": ["teste", "treino"], "taxa_acerto_%": [acc_teste, acc_treino]}).to_csv(
    os.path.join(PASTA, "parte4_resumo.csv"), index=False)

# ---- grafico da fronteira de decisao ----
g = np.linspace(0, 1, 300)
G1, G2 = np.meshgrid(g, g)
Y = rede(np.c_[G1.ravel(), G2.ravel()]).reshape(G1.shape)

fig, ax = plt.subplots(figsize=(8.5, 6))
ax.contourf(G1, G2, Y >= 0, levels=[-.5, .5, 1.5], colors=["#f2d7d5", "#d4e6f1"], alpha=.8)
ax.contour(G1, G2, Y, levels=[0], colors="black", linewidths=1)
for dados_, mk, nome in [(treino, "o", "treino"), (teste, "s", "teste")]:
    for val, cor in [(1, "tab:blue"), (-1, "tab:red")]:
        p = dados_[dados_["d"] == val]
        ax.scatter(p["x1"], p["x2"], marker=mk, c=cor, edgecolors="black",
                   s=40 if nome == "treino" else 80, label=f"{nome} d={val:+d}")
erros = res[~res["acertou"]]
ax.scatter(erros["x1"], erros["x2"], s=260, facecolors="none", edgecolors="black", lw=2, label="erro no teste")
ax.scatter(centros[:, 0], centros[:, 1], marker="*", s=300, c="gold", edgecolors="black", label="centros")
ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x1", ylabel="x2", aspect="equal",
       title=f"Fronteira da RBF (azul = presença) - teste {acc_teste:.0f}%")
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "parte4_fronteira.png"), dpi=150)
print(f"\nArquivos salvos em ./{PASTA}/")
