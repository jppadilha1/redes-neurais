import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Configuracoes do exercicio
# ----------------------------------------------------------------------
TRAINING_FILE = "conjunto_treinamento.csv"
NUM_INPUTS = 3                # x1, x2, x3
NUM_HIDDEN = 10               # neuronios da camada escondida
LEARNING_RATE = 0.1           # eta (dado pelo enunciado)
PRECISION = 1e-6              # epsilon (dado pelo enunciado)
SEEDS = [1, 2, 3, 4, 5]       # uma semente por treinamento (T1 a T5)
MAX_EPOCHS = 100000           # limite de seguranca para nao travar


# ----------------------------------------------------------------------
# 1) Leitura do conjunto de treinamento
# ----------------------------------------------------------------------
def load_data(path):
    """Le o CSV com pandas e devolve:
       - samples: matriz (P x 4) com as colunas [x0, x1, x2, x3], x0 = -1 (bias)
       - targets: vetor com a saida desejada d de cada amostra
    """
    df = pd.read_csv(path)
    df.insert(0, "x0", -1.0)
    samples = df[["x0", "x1", "x2", "x3"]].to_numpy()
    targets = df["d"].to_numpy()
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
    """Fase forward: propaga uma amostra da entrada ate a saida.
       Devolve a saida da camada escondida (com bias) e a saida da rede.
    """
    I1 = W1 @ x                               # soma ponderada dos 10 neuronios escondidos
    Y1 = logistic(I1)                         # saida dos 10 neuronios escondidos
    Y1 = np.concatenate(([-1.0], Y1))         # acrescenta o bias (-1) para a camada de saida
    I2 = W2 @ Y1                              # soma ponderada do neuronio de saida
    Y2 = logistic(I2)                         # saida da rede
    return Y1, Y2


def mean_squared_error(W1, W2, samples, targets):
    """EQM = (1/P) * soma de 1/2 * (d - y)^2 sobre todas as P amostras"""
    total = 0.0
    for x, d in zip(samples, targets):
        _, Y2 = forward(W1, W2, x)
        total += 0.5 * (d - Y2[0]) ** 2
    return total / len(samples)


def train(samples, targets, W1, W2, eta, epsilon, max_epochs):
    """Treina a rede pelo backpropagation (Regra Delta Generalizada).
       Para quando |EQM_atual - EQM_anterior| <= epsilon.
       Devolve (W1, W2, epochs, mse_history, converged).
    """
    W1 = W1.copy()
    W2 = W2.copy()
    mse_history = []
    epoch = 0
    current_mse = mean_squared_error(W1, W2, samples, targets)

    while epoch < max_epochs:
        previous_mse = current_mse

        for x, d in zip(samples, targets):
            # ---- Fase forward ----
            Y1, Y2 = forward(W1, W2, x)

            # ---- Fase backward ----
            # Gradiente local da saida
            delta2 = (d - Y2) * logistic_derivative(Y2)                 # vetor (1,)

            # Gradiente local da escondida (usa W2 ANTES de ser ajustada;
            # a coluna 0 de W2 e o bias, que nao recebe delta)
            delta1 = (delta2 @ W2[:, 1:]) * logistic_derivative(Y1[1:])  # vetor (10,)

            # Ajuste dos pesos:  W <- W + eta * delta * (entrada da camada)
            W2 = W2 + eta * np.outer(delta2, Y1)
            W1 = W1 + eta * np.outer(delta1, x)

        epoch += 1
        current_mse = mean_squared_error(W1, W2, samples, targets)
        mse_history.append(current_mse)

        if abs(current_mse - previous_mse) <= epsilon:   # EQM estabilizou -> terminou
            return W1, W2, epoch, mse_history, True

    return W1, W2, epoch, mse_history, False


# ----------------------------------------------------------------------
# 3) Execucao dos 5 treinamentos
# ----------------------------------------------------------------------
def main():
    samples, targets = load_data(TRAINING_FILE)
    print(f"Conjunto de treinamento: {len(samples)} amostras carregadas.\n")

    rows = []
    histories = []
    weights = {}
    for i, seed in enumerate(SEEDS, start=1):
        name = f"T{i}"
        # Reinicia o gerador com a semente deste treinamento
        rng = np.random.default_rng(seed)
        W1 = rng.random((NUM_HIDDEN, NUM_INPUTS + 1))   # 10 x 4  (bias + 3 entradas)
        W2 = rng.random((1, NUM_HIDDEN + 1))            # 1 x 11  (bias + 10 escondidos)

        print(f"Treinando {name} (semente {seed})...")
        W1, W2, epochs, mse_history, converged = train(
            samples, targets, W1, W2, LEARNING_RATE, PRECISION, MAX_EPOCHS)

        if not converged:
            print(f"ATENCAO: {name} atingiu {MAX_EPOCHS} epocas sem convergir!")

        rows.append({"training": name, "seed": seed,
                     "final_mse": mse_history[-1], "epochs": epochs})
        for epoch, mse in enumerate(mse_history, start=1):
            histories.append({"training": name, "epoch": epoch, "mse": mse})
        weights[f"W1_{name}"] = W1
        weights[f"W2_{name}"] = W2

    # ---- Tabela da Questao 2 ----
    results = pd.DataFrame(rows)
    print()
    print(results.to_string(index=False, formatters={"final_mse": "{:.6f}".format}))

    # ---- Salva os resultados ----
    results.to_csv("training_results_mlp.csv", index=False)
    pd.DataFrame(histories).to_csv("mse_history_mlp.csv", index=False)
    np.savez("mlp_weights.npz", **weights)
    print("\nTabela salva em training_results_mlp.csv")
    print("Historico do EQM salvo em mse_history_mlp.csv")
    print("Pesos finais salvos em mlp_weights.npz")


if __name__ == "__main__":
    main()