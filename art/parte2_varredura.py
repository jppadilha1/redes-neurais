"""PARTE 2 - Varredura da vigilancia (rho de 0,05 a 1,00) para mostrar como o numero de classes cresce.
Nao altera os resultados da parte 1; serve para analisar o efeito do rho.
Gera:
  saidas/parte2_varredura.csv   rho, classes ativas e agrupamentos
  saidas/parte2_varredura.png   numero de classes ativas x rho
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dados import ler, art1, ARQ_TREINO, PASTA, VIGILANCIAS

nomes, X = ler(ARQ_TREINO)
linhas = []
for rho in np.round(np.arange(0.05, 1.0001, 0.01), 2):
    cl, Wb, _, ciclos, _ = art1(X, rho)
    ativos = list(dict.fromkeys(cl))
    renum = {k: i + 1 for i, k in enumerate(ativos)}
    grupos = {}
    for i, k in enumerate(cl):
        grupos.setdefault(renum[k], []).append(nomes[i])
    linhas.append({"rho": rho, "classes_ativas": len(ativos), "neuronios_criados": len(Wb),
                   "agrupamentos": " | ".join(", ".join(g) for g in grupos.values())})
tab = pd.DataFrame(linhas)
tab.to_csv(os.path.join(PASTA, "parte2_varredura.csv"), index=False)

# mostra so onde o agrupamento muda
mudou = tab[tab["agrupamentos"] != tab["agrupamentos"].shift()]
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 120)
print("Faixas de rho com o mesmo agrupamento (a partir de):")
print(mudou[["rho", "classes_ativas", "agrupamentos"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
ax.step(tab["rho"], tab["classes_ativas"], where="post", lw=2)
for r in VIGILANCIAS:
    k = tab.loc[np.isclose(tab["rho"], r), "classes_ativas"].iloc[0]
    ax.plot(r, k, "ro")
    ax.annotate(f"ρ={r}: {k}", (r, k), textcoords="offset points", xytext=(-30, 8), fontsize=8, color="red")
ax.set(xlabel="Vigilância (ρ)", ylabel="Classes ativas", title="Número de classes x vigilância",
       ylim=(0, len(X) + 0.5), yticks=range(0, len(X) + 1))
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "parte2_varredura.png"), dpi=150)
print(f"\nArquivos salvos em ./{PASTA}/")
