import time

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Configuracoes do exercicio
# ----------------------------------------------------------------------
TRAINING_FILE = "treinamento.csv"
INPUT_COLUMNS = ["x1", "x2", "x3", "x4"]
TARGET_COLUMNS = ["d1", "d2", "d3"]
NUM_HIDDEN = 15               # neuronios da camada escondida
LEARNING_RATE = 0.1           # eta (dado pelo enunciado)
MOMENTUM = 0.9                # alpha (dado pelo enunciado)
PRECISION = 1e-6              # epsilon (dado pelo enunciado)
SEED = 1                      # semente dos pesos iniciais (usados nos dois metodos)
MAX_EPOCHS = 100000           # limite de seguranca para nao travar


# ----------------------------------------------------------------------
# 1) Leitura do conjunto de treinamento
# ----------------------------------------------------------------------
def load_data(path):
    """Le o CSV com pandas e devolve:
       - samples: matriz (P x 5) com as colunas [x0, x1, x2, x3, x4], x0 = -1 (bias)
       - targets: matriz (P x 3) com as saidas desejadas [d1, d2, d3]
    """
    df = pd.read_csv(path)
    df.insert(0, "x0", -1.0)
    samples = df[["x0"] + INPUT_COLUMNS].to_numpy(dtype=float)
    targets = df[TARGET_COLUMNS].to_numpy(dtype=float)
    return samples, targets


# ----------------------------------------------------------------------
# 2) Funcoes da rede
# ----------------------------------------------------------------------
def logistic(u):
    """Funcao de ativacao logistica: g(u) = 1 / (1 + e^-u), saida entre 0 e 1"""
    return 1.0 / (1.0 + np.exp(-u))


def logistic_derivative(y):
    """Derivada da logistica escrita em funcao da propria saida: g'(u) = y * (1 - y)"""
    return y * (1.0 - y)


def forward(W1, W2, x):
    """Fase forward: propaga uma amostra da entrada ate as 3 saidas.
       Devolve a saida da camada escondida (com bias) e as saidas da rede.
    """
    Y1 = logistic(W1 @ x)                     # saida dos 15 neuronios escondidos
    Y1 = np.concatenate(([-1.0], Y1))         # acrescenta o bias (-1) para a camada de saida
    Y2 = logistic(W2 @ Y1)                    # saidas y1, y2, y3 da rede
    return Y1, Y2


def mean_squared_error(W1, W2, samples, targets):
    """EQM = (1/P) * soma, sobre as P amostras, de 1/2 * soma (d_j - y_j)^2 nas 3 saidas"""
    total = 0.0
    for x, d in zip(samples, targets):
        _, Y2 = forward(W1, W2, x)
        total += 0.5 * np.sum((d - Y2) ** 2)
    return total / len(samples)


def train(samples, targets, W1, W2, eta, alpha, epsilon, max_epochs):
    """Treina a rede pelo backpropagation.
       - alpha = 0  -> backpropagation padrao
       - alpha > 0  -> backpropagation com momentum:
             W(t+1) = W(t) + alpha * (W(t) - W(t-1)) + eta * delta * entrada
       Para quando |EQM_atual - EQM_anterior| <= epsilon.
       Devolve (W1, W2, epochs, mse_history, converged).
    """
    W1 = W1.copy()
    W2 = W2.copy()
    W1_previous = W1.copy()                   # W(t-1), usado pelo momentum
    W2_previous = W2.copy()
    mse_history = []
    epoch = 0
    current_mse = mean_squared_error(W1, W2, samples, targets)

    while epoch < max_epochs:
        previous_mse = current_mse

        for x, d in zip(samples, targets):
            # ---- Fase forward ----
            Y1, Y2 = forward(W1, W2, x)

            # ---- Fase backward ----
            # Gradiente local dos 3 neuronios de saida
            delta2 = (d - Y2) * logistic_derivative(Y2)                   # vetor (3,)

            # Gradiente local dos 15 neuronios escondidos (usa W2 ANTES do ajuste;
            # a coluna 0 de W2 e o bias, que nao recebe delta)
            delta1 = (delta2 @ W2[:, 1:]) * logistic_derivative(Y1[1:])    # vetor (15,)

            # Ajuste dos pesos com o termo de momentum
            new_W2 = W2 + alpha * (W2 - W2_previous) + eta * np.outer(delta2, Y1)
            new_W1 = W1 + alpha * (W1 - W1_previous) + eta * np.outer(delta1, x)

            W2_previous, W2 = W2, new_W2
            W1_previous, W1 = W1, new_W1

        epoch += 1
        current_mse = mean_squared_error(W1, W2, samples, targets)
        mse_history.append(current_mse)

        if abs(current_mse - previous_mse) <= epsilon:   # EQM estabilizou -> terminou
            return W1, W2, epoch, mse_history, True

    return W1, W2, epoch, mse_history, False


# ----------------------------------------------------------------------
# 3) Execucao dos dois treinamentos
# ----------------------------------------------------------------------
def main():
    samples, targets = load_data(TRAINING_FILE)
    num_inputs = samples.shape[1]             # 4 entradas + bias = 5
    num_outputs = targets.shape[1]            # 3 saidas
    print(f"Conjunto de treinamento: {len(samples)} amostras carregadas.\n")

    # Matrizes de pesos iniciais, sorteadas UMA vez e usadas nos dois metodos
    rng = np.random.default_rng(SEED)
    W1_initial = rng.random((NUM_HIDDEN, num_inputs))         # 15 x 5  (bias + 4 entradas)
    W2_initial = rng.random((num_outputs, NUM_HIDDEN + 1))    # 3 x 16  (bias + 15 escondidos)

    methods = [
        ("Backpropagation padrao", "standard", 0.0),
        ("Backpropagation com momentum", "momentum", MOMENTUM),
    ]

    rows = []
    histories = []
    weights = {}
    for label, key, alpha in methods:
        print(f"Treinando: {label}...")
        start = time.perf_counter()
        W1, W2, epochs, mse_history, converged = train(
            samples, targets, W1_initial, W2_initial,
            LEARNING_RATE, alpha, PRECISION, MAX_EPOCHS)
        elapsed = time.perf_counter() - start

        if not converged:
            print(f"ATENCAO: {label} atingiu {MAX_EPOCHS} epocas sem convergir!")

        rows.append({"method": key, "final_mse": mse_history[-1],
                     "epochs": epochs, "time_seconds": elapsed})
        for epoch, mse in enumerate(mse_history, start=1):
            histories.append({"method": key, "epoch": epoch, "mse": mse})
        weights[f"W1_{key}"] = W1
        weights[f"W2_{key}"] = W2

    # ---- Resultados ----
    results = pd.DataFrame(rows)
    print()
    print(results.to_string(index=False, formatters={
        "final_mse": "{:.6f}".format, "time_seconds": "{:.2f}".format}))

    # ---- Salva os resultados ----
    results.to_csv("training_results_conservante.csv", index=False)
    pd.DataFrame(histories).to_csv("mse_history_conservante.csv", index=False)
    np.savez("mlp_conservante_weights.npz", **weights)
    print("\nResultados salvos em training_results_conservante.csv")
    print("Historico do EQM salvo em mse_history_conservante.csv")
    print("Pesos finais salvos em mlp_conservante_weights.npz")


if __name__ == "__main__":
    main()
