"""PARTE 3 do enunciado - O que acontece quando o ruido aumenta demais.
Para cada nivel de ruido (0% a 100%), corrompe cada padrao 300 vezes e anota onde a rede para:
no padrao certo, em outro padrao, no inverso de um padrao ou num estado espurio.
Gera:
  saidas/parte3_ruido.csv          proporcao de cada resultado por nivel de ruido e padrao
  saidas/parte3_taxa_acerto.png    taxa de recuperacao correta x nivel de ruido
  saidas/parte3_resultados.png     para onde a rede vai em cada nivel de ruido (todos os padroes)
  saidas/parte3_exemplos.png       exemplos de recuperacao com ruido crescente
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dados import ler, treinar, recuperar, identificar, desenhar, ARQ_TREINO, PASTA, SEMENTE, N

TENTATIVAS = 300
NIVEIS = np.arange(0, 101, 5)

meta, Z = ler(ARQ_TREINO)
nomes = meta["padrao"].tolist() if "padrao" in meta else [str(i + 1) for i in range(len(Z))]
W = treinar(Z)
rng = np.random.default_rng(SEMENTE)


def corromper(z, nivel):
    v = z.copy()
    v[rng.choice(N, round(nivel / 100 * N), replace=False)] *= -1
    return v


def categoria(resultado, nome):
    if resultado == f"Padrão {nome}":
        return "Padrão correto"
    if resultado.startswith("Padrão"):
        return "Outro padrão"
    if resultado.startswith("Inverso"):
        return "Inverso de um padrão"
    return "Estado espúrio"


CATS = ["Padrão correto", "Outro padrão", "Inverso de um padrão", "Estado espúrio"]
linhas = []
for nivel in NIVEIS:
    for z, nome in zip(Z, nomes):
        cont = dict.fromkeys(CATS, 0)
        for _ in range(TENTATIVAS):
            v, _, _ = recuperar(corromper(z, nivel), W, rng)
            cont[categoria(identificar(v, Z, nomes), nome)] += 1
        linhas.append({"ruido_%": nivel, "pixels_trocados": round(nivel / 100 * N), "padrao": nome,
                       **{c: cont[c] / TENTATIVAS * 100 for c in CATS}})
    print(f"ruido {nivel:3d}% concluido")

tab = pd.DataFrame(linhas)
tab.to_csv(os.path.join(PASTA, "parte3_ruido.csv"), index=False)
geral = tab.groupby("ruido_%")[CATS].mean()

print("\nMedia dos 4 padroes (% das tentativas):")
print(geral.round(1).to_string())

# ---- taxa de acerto por padrao ----
fig, ax = plt.subplots(figsize=(8, 4.5))
for nome in nomes:
    t = tab[tab["padrao"] == nome]
    ax.plot(t["ruido_%"], t["Padrão correto"], marker="o", ms=3, label=f"Padrão {nome}")
ax.plot(geral.index, geral["Padrão correto"], "k-", lw=2.5, label="Média")
ax.axvline(20, color="gray", ls="--", lw=1)
ax.text(21, 95, "20% (enunciado)", fontsize=8, color="gray")
ax.set(xlabel="Nível de ruído (% de pixels trocados)", ylabel="Recuperação correta (%)",
       title="Taxa de recuperação correta x nível de ruído", ylim=(-3, 103))
ax.grid(alpha=.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "parte3_taxa_acerto.png"), dpi=150)

# ---- para onde a rede vai ----
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.stackplot(geral.index, *[geral[c] for c in CATS], labels=CATS,
             colors=["tab:green", "tab:orange", "tab:purple", "tab:gray"], alpha=.85)
ax.set(xlabel="Nível de ruído (% de pixels trocados)", ylabel="% das tentativas",
       title="Estado final da rede x nível de ruído (média dos 4 padrões)", xlim=(0, 100), ylim=(0, 100))
ax.legend(loc="center left", bbox_to_anchor=(1.01, .5), fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "parte3_resultados.png"), dpi=150)

# ---- exemplos visuais com ruido crescente ----
ex_niveis = [20, 40, 50, 60, 80]
fig, axs = plt.subplots(len(Z), 2 * len(ex_niveis), figsize=(1.45 * 2 * len(ex_niveis), 2.9 * len(Z)))
for lin, (z, nome) in enumerate(zip(Z, nomes)):
    for k, nivel in enumerate(ex_niveis):
        v0 = corromper(z, nivel)
        v, _, _ = recuperar(v0, W, rng)
        desenhar(axs[lin, 2 * k], v0, f"P{nome} – {nivel}%\ndistorcida")
        desenhar(axs[lin, 2 * k + 1], v, f"recuperada\n{identificar(v, Z, nomes).replace('do padrão', 'P')}")
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "parte3_exemplos.png"), dpi=130)
print(f"\nArquivos salvos em ./{PASTA}/")
