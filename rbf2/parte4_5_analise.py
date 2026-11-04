"""PARTES 4 e 5 - Melhor treinamento de cada topologia, graficos e escolha final.
Melhor treinamento de cada rede = menor erro relativo medio no teste (desempate: menor EQM).
Gera:
  saidas/parte4_melhores.csv        melhor treinamento de cada rede (EQM, epocas, erro, variancia)
  saidas/parte4_eqm.png             3 graficos EQM x epoca lado a lado (nao sobrepostos)
  saidas/parte5_comparacao.csv      todos os 9 treinamentos juntos, ordenados pelo erro no teste
  saidas/parte5_desejado_vs_rede.png d e y das 3 melhores redes em cada amostra de teste
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dados import PASTA, TOPOLOGIAS

tab = pd.read_csv(os.path.join(PASTA, "parte2_tabela.csv"))
res = pd.read_csv(os.path.join(PASTA, "parte3_resumo.csv"))
hist = pd.read_csv(os.path.join(PASTA, "parte1_eqm_por_epoca.csv"))
saidas = pd.read_csv(os.path.join(PASTA, "parte3_saidas_teste.csv"))

todos = tab.merge(res, on=["rede", "treinamento"])
todos = todos.sort_values(["erro_relativo_medio_%", "eqm"]).reset_index(drop=True)
todos.to_csv(os.path.join(PASTA, "parte5_comparacao.csv"), index=False)

melhores = todos.groupby("rede", sort=False).head(1).set_index("rede").loc[list(TOPOLOGIAS)].reset_index()
melhores.to_csv(os.path.join(PASTA, "parte4_melhores.csv"), index=False)

# ---- parte 4: EQM x epoca, 3 graficos lado a lado ----
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (_, m) in zip(axs, melhores.iterrows()):
    h = hist[(hist["rede"] == m["rede"]) & (hist["treinamento"] == m["treinamento"])]
    ax.plot(h["epoca"], h["eqm"])
    ax.set(title=f"{m['rede']} (N1 = {m['N1']}) - {m['treinamento']}\n"
                 f"{m['epocas']} épocas | EQM final = {m['eqm']:.6f}",
           xlabel="Época", ylabel="EQM")
    ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "parte4_eqm.png"), dpi=150)

# ---- parte 5: desejado x saida das 3 melhores ----
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(saidas["amostra"], saidas["d"], "k-o", lw=2, label="desejado (d)")
for _, m in melhores.iterrows():
    col = f"{m['rede']} y({m['treinamento']})"
    ax.plot(saidas["amostra"], saidas[col], "--o", ms=4,
            label=f"{m['rede']} {m['treinamento']} (erro {m['erro_relativo_medio_%']:.2f}%)")
ax.set(xlabel="Amostra de teste", ylabel="y", xticks=saidas["amostra"],
       title="Saída desejada x saída das melhores redes")
ax.grid(alpha=.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "parte5_desejado_vs_rede.png"), dpi=150)

pd.set_option("display.width", 200)
print("Melhor treinamento de cada topologia:")
print(melhores.round(6).to_string(index=False))
print("\nTodos os treinamentos (ordenados pelo erro relativo medio no teste):")
print(todos.round(6).to_string(index=False))
m = todos.iloc[0]
print(f"\nMenor erro no teste: {m['rede']} (N1 = {m['N1']}) com {m['treinamento']} "
      f"-> {m['erro_relativo_medio_%']:.4f}% (variancia {m['variancia_%']:.4f})")
print(f"\nArquivos salvos em ./{PASTA}/")
