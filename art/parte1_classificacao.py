"""PARTE 1 - Treinamento/classificacao da ART-1 para rho = 0.5, 0.8, 0.9 e 0.99.
Para cada vigilancia: quantas classes ficaram ativas e quais situacoes estao em cada uma.
Classe ativa = neuronio que terminou o treinamento com pelo menos uma situacao.
Gera:
  saidas/parte1_resumo.csv                 rho, classes ativas, neuronios criados, ciclos, agrupamentos
  saidas/parte1_classes.csv                classe de cada situacao para cada rho
  saidas/parte1_prototipos_rho_<rho>.csv   pesos top-down (padrao representativo) de cada classe ativa
  saidas/parte1_primeiro_ciclo.csv         classe de cada situacao ja na 1a apresentacao (antes de estabilizar)
  saidas/parte1_agrupamentos.png           situacoes agrupadas por classe para cada rho
  saidas/parte1_mapa_classes.png           tabela colorida: situacao x rho -> classe
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dados import ler, art1, ARQ_TREINO, PASTA, VIGILANCIAS

nomes, X = ler(ARQ_TREINO)
print(f"Situacoes: {len(X)} | variaveis: {X.shape[1]}")
iguais = [(nomes[a], nomes[b]) for a in range(len(X)) for b in range(a + 1, len(X)) if np.array_equal(X[a], X[b])]
if iguais:
    print("Situacoes identicas:", ", ".join(f"{a}={b}" for a, b in iguais))

resumo, tab_classes, prim = [], pd.DataFrame({"situacao": nomes}), pd.DataFrame({"situacao": nomes})
resultados = {}
for rho in VIGILANCIAS:
    cl, Wb, Wf, ciclos, hist = art1(X, rho)
    ativos = list(dict.fromkeys(cl))                     # neuronios com situacoes, na ordem em que aparecem
    renum = {k: i + 1 for i, k in enumerate(ativos)}    # classes ativas numeradas 1, 2, 3...
    classe = np.array([renum[k] for k in cl])
    grupos = {c: [nomes[i] for i in np.where(classe == c)[0]] for c in range(1, len(ativos) + 1)}
    resultados[rho] = (classe, Wb[ativos], grupos)

    tab_classes[f"rho_{rho}"] = classe
    h1 = pd.DataFrame(hist)
    prim[f"rho_{rho}"] = h1[h1["ciclo"] == 1].sort_values("situacao")["neuronio"].values

    proto = pd.DataFrame(Wb[ativos].astype(int), columns=[f"x{i}" for i in range(1, X.shape[1] + 1)])
    proto.insert(0, "situacoes", [", ".join(grupos[c]) for c in grupos])
    proto.insert(0, "classe", list(grupos))
    proto.to_csv(os.path.join(PASTA, f"parte1_prototipos_rho_{rho}.csv"), index=False)

    resumo.append({"rho": rho, "classes_ativas": len(ativos), "neuronios_criados": len(Wb),
                   "ciclos_ate_estabilizar": ciclos,
                   "agrupamentos": " | ".join(f"Classe {c}: {', '.join(g)}" for c, g in grupos.items())})
    print(f"\nrho = {rho}: {len(ativos)} classes ativas ({len(Wb)} neuronios criados, {ciclos} ciclos)")
    for c, g in grupos.items():
        print(f"   Classe {c}: {', '.join(g)}")

pd.DataFrame(resumo).to_csv(os.path.join(PASTA, "parte1_resumo.csv"), index=False)
tab_classes.to_csv(os.path.join(PASTA, "parte1_classes.csv"), index=False)
prim.to_csv(os.path.join(PASTA, "parte1_primeiro_ciclo.csv"), index=False)

# ---- figura: situacoes agrupadas por classe ----
fig, axs = plt.subplots(1, len(VIGILANCIAS), figsize=(5.2 * len(VIGILANCIAS), 5.2))
for ax, rho in zip(axs, VIGILANCIAS):
    classe, proto, grupos = resultados[rho]
    ordem = np.argsort(classe, kind="stable")
    ax.imshow(X[ordem], cmap="Greys", vmin=-0.3, vmax=1.2, aspect="auto")
    rot = [f"{nomes[i]} (C{classe[i]})" for i in ordem]
    ax.set_yticks(range(len(ordem)), rot, fontsize=8)
    ax.set_xticks(range(X.shape[1]), [f"x{i}" for i in range(1, X.shape[1] + 1)], fontsize=6, rotation=90)
    for lim in np.where(np.diff(classe[ordem]) != 0)[0]:
        ax.axhline(lim + 0.5, color="red", lw=2)
    ax.set_title(f"ρ = {rho}: {len(grupos)} classes", fontsize=11)
fig.suptitle("Situações agrupadas por classe (preto = 1; linhas vermelhas separam as classes)")
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "parte1_agrupamentos.png"), dpi=150)

# ---- figura: mapa situacao x rho ----
M = tab_classes[[f"rho_{r}" for r in VIGILANCIAS]].values
fig, ax = plt.subplots(figsize=(6, 5.5))
ax.imshow(M, cmap="tab10", vmin=1, vmax=10, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"C{M[i, j]}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
ax.set_xticks(range(len(VIGILANCIAS)), [f"ρ = {r}\n({len(resultados[r][2])} classes)" for r in VIGILANCIAS])
ax.set_yticks(range(len(nomes)), nomes)
ax.set_title("Classe de cada situação para cada vigilância")
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "parte1_mapa_classes.png"), dpi=150)
print(f"\nArquivos salvos em ./{PASTA}/")
