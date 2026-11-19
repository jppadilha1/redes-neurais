"""Funcoes comuns da rede ART-1 (16 entradas binarias).
Usado por todos os scripts. Nao precisa rodar este arquivo.

Algoritmo (Silva, Spatti e Flauzino - Redes Neurais Artificiais, cap. 11):
  - n = 16 entradas; neuronios de saida criados conforme necessario.
  - Pesos iniciais de um neuronio novo:
        top-down  (W_b): todos 1
        bottom-up (W_f): todos 1 / (1 + n)
  - Para cada entrada x:
        1) ativacao de cada neuronio habilitado: u_j = W_f[j] . x
        2) vencedor k = maior u_j
        3) teste de vigilancia: R = (W_b[k] . x) / (soma de x)
             R >= rho -> ressonancia: o vencedor aprende x
             R <  rho -> desabilita k e volta ao passo 2
           se nenhum neuronio passar, cria um neuronio novo (classe nova)
        4) aprendizado do vencedor k:
             W_b[k] <- W_b[k] AND x
             W_f[k] <- (W_b[k] AND x) / (0,5 + soma(W_b[k] AND x))
  - As situacoes sao apresentadas em ordem; o ciclo e repetido ate nenhuma
    situacao mudar de classe e nenhum peso mudar (estabilidade).
"""
import os
import re
import numpy as np
import pandas as pd

ARQ_TREINO = "treinamento.csv"
PASTA = "saidas"
os.makedirs(PASTA, exist_ok=True)

VIGILANCIAS = [0.5, 0.8, 0.9, 0.99]
MAX_CICLOS = 100


def ler(caminho):
    """Devolve (lista de nomes das situacoes, matriz binaria k x 16).
    Usa as colunas x1..x16 se existirem; senao, as 16 ultimas colunas.
    A primeira coluna nao numerica (ex.: 'Situacao 1') vira o nome da situacao."""
    df = pd.read_csv(caminho, sep=None, engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]
    xs = [c for c in df.columns if re.fullmatch(r"x\s*\d+", c)]
    if len(xs) != 16:
        xs = list(df.columns[-16:])
    X = df[xs].apply(pd.to_numeric).values.astype(int)
    outras = [c for c in df.columns if c not in xs]
    if outras:
        ids = [re.sub(r"\D", "", str(v)) or str(v) for v in df[outras[0]]]
    else:
        ids = [str(i) for i in range(1, len(df) + 1)]
    return [f"S{i}" for i in ids], (X > 0).astype(int)


def art1(X, rho, max_ciclos=MAX_CICLOS):
    """Treina a ART-1 com vigilancia rho. Devolve (classe de cada situacao, W_b, W_f, ciclos, historico)."""
    n = X.shape[1]
    Wb = np.zeros((0, n))
    Wf = np.zeros((0, n))
    classe = np.full(len(X), -1)
    historico = []
    for ciclo in range(1, max_ciclos + 1):
        Wb_ant, classe_ant = Wb.copy(), classe.copy()
        for i, x in enumerate(X):
            habilitado = np.ones(len(Wb), bool)
            k = -1
            while habilitado.any():
                u = np.where(habilitado, Wf @ x, -np.inf)
                j = int(np.argmax(u))                       # vencedor (empate: menor indice)
                R = (Wb[j] @ x) / x.sum()
                if R >= rho:
                    k = j
                    break
                habilitado[j] = False                       # reset: falhou na vigilancia
            if k == -1:                                     # nenhum passou: cria neuronio novo
                Wb = np.vstack([Wb, np.ones(n)])
                Wf = np.vstack([Wf, np.full(n, 1 / (1 + n))])
                k = len(Wb) - 1
            novo = Wb[k] * x
            Wb[k] = novo
            Wf[k] = novo / (0.5 + novo.sum())
            classe[i] = k
            historico.append({"ciclo": ciclo, "situacao": i, "neuronio": k + 1})
        if ciclo > 1 and np.array_equal(classe, classe_ant) and Wb.shape == Wb_ant.shape \
                and np.array_equal(Wb, Wb_ant):
            break
    return classe, Wb, Wf, ciclo, historico
