"""PARTE 3 - Validacao dos 9 treinamentos com o conjunto de teste.
Erro relativo de cada amostra (%) = |d - y| / |d| * 100
Erro relativo medio (%) = media desses erros
Variancia (%)           = variancia (populacional) desses erros em %
Gera:
  saidas/parte3_saidas_teste.csv   tabela do enunciado: amostra, x1, x2, x3, d e y de cada rede/treinamento
  saidas/parte3_erros_amostra.csv  erro relativo (%) de cada amostra em cada treinamento
  saidas/parte3_resumo.csv         erro relativo medio (%) e variancia (%) de cada treinamento
"""
import os
import numpy as np
import pandas as pd
from dados import ler, entrada_saida, carregar_modelo, ARQ_TESTE, PASTA, ENTRADAS, TOPOLOGIAS, SEMENTES

teste = ler(ARQ_TESTE)
X, d = teste[ENTRADAS].values, teste["d"].values

saidas = teste.copy()
saidas.insert(0, "amostra", np.arange(1, len(teste) + 1))
erros = saidas[["amostra"]].copy()
resumo = []
for rede in TOPOLOGIAS:
    for t in SEMENTES:
        c, v, W = carregar_modelo(rede, t)
        y = entrada_saida(X, c, v) @ W
        er = np.abs(d - y) / np.abs(d) * 100
        col = f"{rede} y({t})"
        saidas[col] = y
        erros[col] = er
        resumo.append({"rede": rede, "treinamento": t,
                       "erro_relativo_medio_%": er.mean(), "variancia_%": er.var()})

saidas.to_csv(os.path.join(PASTA, "parte3_saidas_teste.csv"), index=False)
erros.to_csv(os.path.join(PASTA, "parte3_erros_amostra.csv"), index=False)
resumo = pd.DataFrame(resumo)
resumo.to_csv(os.path.join(PASTA, "parte3_resumo.csv"), index=False)

pd.set_option("display.width", 250)
print(saidas.round(4).to_string(index=False))
print("\nErro relativo medio e variancia:")
print(resumo.round(4).to_string(index=False))
print(f"\nArquivos salvos em ./{PASTA}/")
