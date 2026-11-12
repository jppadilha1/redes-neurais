"""PARTES 1 e 2 do enunciado - 12 situacoes de transmissao (3 por padrao, ~20% de ruido).
Le as imagens distorcidas de teste.csv e recupera cada uma com a rede de Hopfield.
Gera:
  saidas/parte2_resultados.csv        resumo de cada situacao
  saidas/parte2_recuperadas.csv       imagem final de cada situacao (45 pixels)
  saidas/parte2_padrao_<n>.png        transmitida / distorcida / recuperada (3 situacoes por padrao)
  saidas/parte2_energia.png           energia da rede a cada iteracao nas 12 situacoes
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dados import (ler, treinar, recuperar, identificar, desenhar, ARQ_TREINO, ARQ_TESTE,
                   PASTA, SEMENTE, N)


def mais_parecido(v, Z, nomes):
    """Padrao armazenado com mais pixels em comum com a imagem v (usado quando o teste nao diz a origem)."""
    return nomes[int(np.argmax(Z @ v))]

meta_tr, Z = ler(ARQ_TREINO)
nomes = meta_tr["padrao"].tolist() if "padrao" in meta_tr else [str(i + 1) for i in range(len(Z))]
W = treinar(Z)
original = dict(zip(nomes, Z))

meta, X = ler(ARQ_TESTE)
print(f"Imagens lidas de {ARQ_TESTE}: {len(X)}")
if "situacao" not in meta:
    meta["situacao"] = np.arange(1, len(X) + 1)
if "padrao" not in meta or meta["padrao"].isin(nomes).sum() < len(meta):
    # o teste.csv nao informa de qual padrao veio cada imagem: usa o padrao mais parecido
    meta["padrao"] = [mais_parecido(v, Z, nomes) for v in X]
    print("Coluna 'padrao' ausente no teste.csv -> origem de cada imagem = padrao mais parecido")

rng = np.random.default_rng(SEMENTE)
linhas, finais, energias = [], [], []
for (_, m), v0 in zip(meta.iterrows(), X):
    z = original[m["padrao"]]
    v, it, en = recuperar(v0, W, rng)
    finais.append(v)
    energias.append(en)
    linhas.append({
        "situacao": m["situacao"], "padrao": m["padrao"],
        "pixels_corrompidos": int((v0 != z).sum()),
        "ruido_%": round((v0 != z).mean() * 100, 1),
        "iteracoes": it,
        "resultado": identificar(v, Z, nomes),
        "recuperou_corretamente": bool(np.array_equal(v, z)),
        "pixels_errados_no_final": int((v != z).sum()),
    })
res = pd.DataFrame(linhas)
res.to_csv(os.path.join(PASTA, "parte2_resultados.csv"), index=False)
rec = pd.DataFrame(finais, columns=[f"p{i:02d}" for i in range(1, N + 1)])
rec.insert(0, "padrao", res["padrao"])
rec.insert(0, "situacao", res["situacao"])
rec.to_csv(os.path.join(PASTA, "parte2_recuperadas.csv"), index=False)

pd.set_option("display.width", 200)
print(res.to_string(index=False))
print(f"\nRecuperadas corretamente: {res['recuperou_corretamente'].sum()} de {len(res)}")

# ---- figuras: uma por padrao (linhas = situacoes; colunas = transmitida, distorcida, recuperada) ----
for nome in nomes:
    idx = res.index[res["padrao"] == nome]
    fig, axs = plt.subplots(len(idx), 3, figsize=(5.4, 3.3 * len(idx)), squeeze=False)
    for lin, i in enumerate(idx):
        r = res.loc[i]
        desenhar(axs[lin, 0], original[nome], f"Situação {r['situacao']}\nTransmitida")
        desenhar(axs[lin, 1], X[i], f"Distorcida\n({r['pixels_corrompidos']} pixels trocados)")
        marca = "✔" if r["recuperou_corretamente"] else "✘"
        desenhar(axs[lin, 2], finais[i], f"Recuperada {marca}\n({r['iteracoes']} iterações)")
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA, f"parte2_padrao_{nome}.png"), dpi=150)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
for r, en in zip(linhas, energias):
    ax.plot(en, marker="o", ms=3, label=f"S{r['situacao']} (padrão {r['padrao']})")
ax.set(xlabel="Iteração", ylabel="Energia", title="Energia da rede durante a recuperação")
ax.grid(alpha=.3)
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "parte2_energia.png"), dpi=150)
print(f"\nArquivos salvos em ./{PASTA}/")
