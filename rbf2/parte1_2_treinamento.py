"""PARTES 1 e 2 - Treinamento das 3 topologias (N1 = 5, 10, 15), 3 vezes cada (T1, T2, T3).
Camada escondida: k-means com TODAS as amostras de treino (aproximacao de funcao, nao ha classes).
Camada de saida: regra delta, eta = 0.01, eps = 1e-7, pesos iniciais aleatorios em [0, 1]
com uma semente diferente em cada treinamento.
Gera:
  saidas/parte1_centros.csv          centros e variancias de cada rede
  saidas/parte1_pesos_iniciais.csv   pesos sorteados antes de cada treinamento (prova que sao diferentes)
  saidas/parte1_pesos_finais.csv     pesos apos a convergencia
  saidas/parte1_eqm_por_epoca.csv    curva do EQM de cada treinamento (usada na parte 4)
  saidas/parte2_tabela.csv           tabela da questao 2: EQM final e numero de epocas
"""
import os
import time
import numpy as np
import pandas as pd
from dados import (ler, kmeans, entrada_saida, regra_delta, ARQ_TREINO, PASTA,
                   ENTRADAS, TOPOLOGIAS, SEMENTES)

treino = ler(ARQ_TREINO)
X, d = treino[ENTRADAS].values, treino["d"].values
print(f"Amostras de treino: {len(X)}\n")

centros_l, ini_l, fin_l, hist_l, tab_l = [], [], [], [], []
for rede, n1 in TOPOLOGIAS.items():
    c, v, r = kmeans(X, n1)
    for i in range(n1):
        centros_l.append({"rede": rede, "neuronio": i + 1, "x1": c[i, 0], "x2": c[i, 1], "x3": c[i, 2],
                          "variancia": v[i], "n_amostras": int((r == i).sum())})
    Z = entrada_saida(X, c, v)

    for t, semente in SEMENTES.items():
        inicio = time.time()
        W0, W, hist = regra_delta(Z, d, semente)
        epocas = len(hist) - 1
        print(f"{rede} (N1={n1:2d}) {t}: EQM = {hist[-1]:.6f} | epocas = {epocas:6d} | {time.time()-inicio:.1f}s")
        for j in range(len(W)):
            ini_l.append({"rede": rede, "treinamento": t, "indice": j, "peso": f"W2_1,{j}", "valor": W0[j]})
            fin_l.append({"rede": rede, "treinamento": t, "indice": j, "peso": f"W2_1,{j}", "valor": W[j]})
        hist_l.append(pd.DataFrame({"rede": rede, "treinamento": t, "epoca": np.arange(len(hist)), "eqm": hist}))
        tab_l.append({"rede": rede, "N1": n1, "treinamento": t, "semente": semente, "eqm": hist[-1], "epocas": epocas})
    print()

pd.DataFrame(centros_l).to_csv(os.path.join(PASTA, "parte1_centros.csv"), index=False)
pd.DataFrame(ini_l).to_csv(os.path.join(PASTA, "parte1_pesos_iniciais.csv"), index=False)
pd.DataFrame(fin_l).to_csv(os.path.join(PASTA, "parte1_pesos_finais.csv"), index=False)
pd.concat(hist_l).to_csv(os.path.join(PASTA, "parte1_eqm_por_epoca.csv"), index=False)
tab = pd.DataFrame(tab_l)
tab.to_csv(os.path.join(PASTA, "parte2_tabela.csv"), index=False)

# tabela no formato do enunciado (linhas T1..T3, colunas Rede x {EQM, Epocas})
print("Tabela da questao 2:")
print(f"{'':12s}" + "".join(f"{r:^24s}" for r in TOPOLOGIAS))
print(f"{'Treinamento':12s}" + "".join(f"{'EQM':>12s}{'Epocas':>12s}" for _ in TOPOLOGIAS))
for t in SEMENTES:
    linha = f"{t:12s}"
    for r in TOPOLOGIAS:
        x = tab[(tab["rede"] == r) & (tab["treinamento"] == t)].iloc[0]
        linha += f"{x['eqm']:>12.6f}{int(x['epocas']):>12d}"
    print(linha)
print(f"\nArquivos salvos em ./{PASTA}/")
