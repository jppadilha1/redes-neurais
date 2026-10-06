"""A diferença entre o ADALINE e o Perceptron ( adaline é aceita ruídos)
é que o Perceptron ajusta os pesos usando o resultado pós degrau (y)funcao ativação e aplica o hebb se y != d

já o adaline ele não passa pela função de ativação na fase de treinamento
ele faz d - u e se não for zero aplica a regra de Delta

para todas as amostras em cada treinamento, ele calcula o EQM (erro quadrático médio) 
e só para quando EQM de antes - EQM de agora <= precisão do fornecida no enunciado
"""

import numpy as np
import pandas as pd


TRAINING_FILE = "adaline_treinamento.csv"
LEARNING_RATE = 0.0025        # eta (dado pelo enunciado)
PRECISION = 1e-6              # epsilon (dado pelo enunciado)
SEEDS = [1, 2, 3, 4, 5]       # uma semente por treinamento (T1 a T5)
MAX_EPOCHS = 100000           # limite de seguranca para nao travar


# ----------------------------------------------------------------------
# 1) Leitura do conjunto de treinamento
# ----------------------------------------------------------------------
def load_data(path):
    """Le o CSV com pandas e devolve:
       - samples: matriz (35 x 5) com as colunas [x0, x1, x2, x3, x4], x0 = -1
       - targets: vetor com a saida desejada d de cada amostra (-1 ou +1)
    """
    df = pd.read_csv(path)
    df.insert(0, "x0", -1.0)                       # entrada fixa do limiar
    samples = df[["x0", "x1", "x2", "x3", "x4"]].to_numpy()
    targets = df["d"].to_numpy()
    return samples, targets


# ----------------------------------------------------------------------
# 2) Funcoes do ADALINE
# ----------------------------------------------------------------------
def mean_squared_error(w, samples, targets):
    """EQM = (1/p) * soma de (d - u)^2 sobre todas as p amostras.
       Repare que o erro usa u (antes do degrau), e nao y.
    """
    u = samples @ w                                # u de todas as amostras de uma vez
    return np.mean((targets - u) ** 2)


def train(samples, targets, initial_weights, eta, epsilon, max_epochs):
    """Treina o ADALINE pela Regra Delta.
       A cada epoca, ajusta os pesos amostra por amostra e calcula o EQM.
       Para quando o EQM praticamente nao muda mais:
           |EQM_atual - EQM_anterior| <= epsilon
       Devolve (final_weights, epochs, mse_history, converged).
    """
    w = initial_weights.copy()                     # copia para nao alterar o original
    mse_history = []
    epoch = 0
    current_mse = mean_squared_error(w, samples, targets)

    while epoch < max_epochs:
        previous_mse = current_mse

        for x, d in zip(samples, targets):
            u = np.dot(w, x)                       # u = w0*x0 + w1*x1 + ... + w4*x4
            # Regra Delta:  w <- w + eta * (d - u) * x
            # (corrige SEMPRE, proporcional ao tamanho do erro)
            w = w + eta * (d - u) * x

        epoch += 1
        current_mse = mean_squared_error(w, samples, targets)
        mse_history.append(current_mse)

        if abs(current_mse - previous_mse) <= epsilon:   # EQM estabilizou -> terminou
            return w, epoch, mse_history, True

    return w, epoch, mse_history, False            # atingiu o limite sem convergir


# ----------------------------------------------------------------------
# 3) Execucao dos 5 treinamentos
# ----------------------------------------------------------------------
def main():
    samples, targets = load_data(TRAINING_FILE)
    print(f"Conjunto de treinamento: {len(samples)} amostras carregadas.\n")

    rows = []
    histories = []
    for i, seed in enumerate(SEEDS, start=1):
        # Reinicia o gerador com a semente deste treinamento
        rng = np.random.default_rng(seed)
        initial_weights = rng.random(5)            # 5 valores em [0, 1): w0 a w4

        final_weights, epochs, mse_history, converged = train(
            samples, targets, initial_weights, LEARNING_RATE, PRECISION, MAX_EPOCHS)

        if not converged:
            print(f"ATENCAO: T{i} atingiu {MAX_EPOCHS} epocas sem convergir!")

        row = {"training": f"T{i}", "seed": seed}
        for j in range(5):
            row[f"w{j}_init"] = initial_weights[j]
        for j in range(5):
            row[f"w{j}_final"] = final_weights[j]
        row["epochs"] = epochs
        row["final_mse"] = mse_history[-1]
        rows.append(row)

        for epoch, mse in enumerate(mse_history, start=1):
            histories.append({"training": f"T{i}", "epoch": epoch, "mse": mse})

    # ---- Tabela da Questao 2 ----
    results = pd.DataFrame(rows).round(4)
    print(results.to_string(index=False))

    # ---- Salva os resultados ----
    results.to_csv("training_results_adaline.csv", index=False)
    pd.DataFrame(histories).to_csv("mse_history.csv", index=False)
    print("\nTabela salva em training_results_adaline.csv")
    print("Historico do EQM salvo em mse_history.csv")


if __name__ == "__main__":
    main()