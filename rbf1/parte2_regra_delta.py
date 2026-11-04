"""PARTE 2 - Treinamento da camada de saida com a regra delta generalizada.
Le os centros/variancias da parte 1 e usa TODAS as amostras de treino.
eta = 0.01, epsilon = 1e-7 (para quando |EQM_atual - EQM_anterior| <= epsilon).
Gera:
  saidas/parte2_pesos.csv         W2_1,0  W2_1,1  W2_1,2
  saidas/parte2_eqm_por_epoca.csv curva do erro
  saidas/parte2_treino_phi.csv    phi1, phi2 e y de cada amostra de treino
  saidas/parte2_eqm.png           grafico do erro por epoca
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dados import ler, entrada_saida, carregar_centros, ARQ_TREINO, PASTA

ETA, EPS, SEMENTE, MAX_EPOCAS = 0.01, 1e-7, 0, 100_000

treino = ler(ARQ_TREINO)
X, d = treino[["x1", "x2"]].values, treino["d"].values
centros, variancias = carregar_centros()
Z = entrada_saida(X, centros, variancias)          # cada linha: [-1, phi1, phi2]


def eqm(W):  # erro quadratico medio
    return 0.5 * np.mean((d - Z @ W) ** 2)


rng = np.random.default_rng(SEMENTE)
W = rng.uniform(0, 1, 3)                           # pesos iniciais aleatorios
print("Pesos iniciais:", np.round(W, 4))

historico = [eqm(W)]
for epoca in range(1, MAX_EPOCAS + 1):
    for z, dk in zip(Z, d):                        # ajuste amostra a amostra
        y = z @ W
        W = W + ETA * (dk - y) * z                 # regra delta
    historico.append(eqm(W))
    if abs(historico[-1] - historico[-2]) <= EPS:
        break

print(f"Convergiu em {epoca} epocas | EQM final = {historico[-1]:.6f}")
pesos = pd.DataFrame({"peso": ["W2_1,0", "W2_1,1", "W2_1,2"], "valor": W})
print(pesos.round(4).to_string(index=False))

pesos.to_csv(os.path.join(PASTA, "parte2_pesos.csv"), index=False)
pd.DataFrame({"epoca": range(len(historico)), "eqm": historico}).to_csv(
    os.path.join(PASTA, "parte2_eqm_por_epoca.csv"), index=False)
saida = treino.copy()
saida["phi1"], saida["phi2"], saida["y"] = Z[:, 1], Z[:, 2], Z @ W
saida.to_csv(os.path.join(PASTA, "parte2_treino_phi.csv"), index_label="amostra")

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(historico)
ax.set(xlabel="Época", ylabel="EQM", title=f"Regra delta (η = {ETA}, ε = {EPS:g}) - {epoca} épocas")
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "parte2_eqm.png"), dpi=150)
print(f"\nArquivos salvos em ./{PASTA}/")
