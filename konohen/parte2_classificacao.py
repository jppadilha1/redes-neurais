"""PARTE 2 - Classificacao das 12 amostras de teste (questao 2).
Cada amostra e apresentada a rede treinada; a classe e a do neuronio vencedor.
Se o vencedor nao venceu nenhuma amostra no treino (sem classe), usa o neuronio com classe mais proximo no grid.
Como conferencia, tambem calcula a distancia da amostra ao centro (media) de cada classe do treino.
Gera:
  saidas/parte2_classificacao.csv   amostra, x1, x2, x3, neuronio vencedor, classe, distancias aos centros
  saidas/parte2_grid_teste.png      em que neuronio do grid caiu cada amostra de teste
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dados import (ler, classe_da_amostra, vencedor, desenhar_grid, carregar_pesos, carregar_neuronios,
                   ARQ_TREINO, ARQ_TESTE, PASTA, ENTRADAS, N1, POS, CLASSES)

W = carregar_pesos()
neur = carregar_neuronios()
classe_neur = dict(zip(neur["neuronio"], neur["classe"]))
com_classe = neur.loc[neur["classe"] != "-", "neuronio"].values

treino = ler(ARQ_TREINO)
treino["classe"] = treino["amostra"].map(classe_da_amostra)
centros = treino.groupby("classe")[ENTRADAS].mean().loc[list(CLASSES)]

teste = ler(ARQ_TESTE)
linhas = []
for r in teste.itertuples():
    x = np.array([r.x1, r.x2, r.x3])
    j = vencedor(x, W) + 1
    classe, obs = classe_neur[j], ""
    if classe == "-":   # neuronio sem classe: pega o neuronio rotulado mais proximo no grid
        k = com_classe[np.argmin(np.abs(POS[com_classe - 1] - POS[j - 1]).sum(axis=1))]
        classe, obs = classe_neur[k], f"vencedor sem classe; vizinho {k}"
    dist = np.sqrt(((centros.values - x) ** 2).sum(axis=1))
    linhas.append({"amostra": r.amostra, "x1": r.x1, "x2": r.x2, "x3": r.x3, "neuronio_vencedor": j,
                   "classe": classe, **{f"dist_centro_{c}": d for c, d in zip(CLASSES, dist)},
                   "classe_centro_mais_proximo": list(CLASSES)[int(np.argmin(dist))], "obs": obs})
res = pd.DataFrame(linhas)
res.to_csv(os.path.join(PASTA, "parte2_classificacao.csv"), index=False)

pd.set_option("display.width", 200)
print(res.round(4).to_string(index=False))
conf = (res["classe"] == res["classe_centro_mais_proximo"]).sum()
print(f"\nClasse pela rede = classe do centro mais proximo em {conf} de {len(res)} amostras")
print("\nCentros das classes no treino:")
print(centros.round(4).to_string())

# ---- grid com as amostras de teste ----
fig, ax = plt.subplots(figsize=(6, 6))
por_neur = res.groupby("neuronio_vencedor")["amostra"].apply(lambda s: ", ".join(map(str, s))).to_dict()
textos = [f"teste: {por_neur[j]}" if j in por_neur else "" for j in range(1, N1 + 1)]
desenhar_grid(ax, neur["classe"].tolist(), textos, "Neurônio vencedor de cada amostra de teste")
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "parte2_grid_teste.png"), dpi=150)
print(f"\nArquivos salvos em ./{PASTA}/")
