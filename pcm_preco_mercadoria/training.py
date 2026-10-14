import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Configuracoes do exercicio
# ----------------------------------------------------------------------
TRAINING_FILE = "treinamento.csv"
VALUE_COLUMN = "f(t)"         # coluna com os valores da serie
NETWORKS = {                  # nome: (p = numero de entradas, N1 = neuronios escondidos)
    "Rede 1": (5, 10),
    "Rede 2": (10, 15),
    "Rede 3": (15, 25),
}
SEEDS = [1, 2, 3]             # uma semente por treinamento (T1, T2, T3)
LEARNING_RATE = 0.1           # eta
MOMENTUM = 0.8                # alpha
PRECISION = 0.5e-6            # epsilon
MAX_EPOCHS = 100000           # limite de seguranca para nao travar


# ----------------------------------------------------------------------
# 1) Leitura da serie e montagem dos padroes (janela de tempo)
# ----------------------------------------------------------------------
def load_series(path):
    """Le o CSV e devolve os valores da serie f(t) como vetor."""
    df = pd.read_csv(path)
    column = VALUE_COLUMN if VALUE_COLUMN in df.columns else df.columns[-1]
    return df[column].to_numpy(dtype=float)


def build_patterns(series, p):
    """Monta os padroes da TDNN com uma janela deslizante de tamanho p.
       Para cada instante t (de p+1 ate o fim da serie):
           entradas = [x(t-1), x(t-2), ..., x(t-p)]
           desejado = f(t)
       Devolve:
         - samples: matriz com [-1 (bias), x(t-1), ..., x(t-p)] em cada linha
         - targets: vetor com f(t)
    """
    samples = []
    targets = []
    for t in range(p, len(series)):           # indice 0 da serie = t = 1
        window = series[t - p:t][::-1]         # x(t-1), x(t-2), ..., x(t-p)
        samples.append(np.concatenate(([-1.0], window)))
        targets.append(series[t])
    return np.array(samples), np.array(targets)


# ----------------------------------------------------------------------
# 2) Funcoes da rede
# ----------------------------------------------------------------------
def logistic(u):
    """Funcao de ativacao logistica: g(u) = 1 / (1 + e^-u)"""
    return 1.0 / (1.0 + np.exp(-u))


def logistic_derivative(y):
    """Derivada da logistica em funcao da propria saida: g'(u) = y * (1 - y)"""
    return y * (1.0 - y)


def forward(W1, W2, x):
    """Fase forward: entrada -> camada escondida -> saida."""
    Y1 = logistic(W1 @ x)                     # saida dos N1 neuronios escondidos
    Y1 = np.concatenate(([-1.0], Y1))         # bias para a camada de saida
    Y2 = logistic(W2 @ Y1)                    # saida da rede, f(t) estimado
    return Y1, Y2


def mean_squared_error(W1, W2, samples, targets):
    """EQM = (1/P) * soma de 1/2 * (d - y)^2 sobre todos os P padroes"""
    total = 0.0
    for x, d in zip(samples, targets):
        _, Y2 = forward(W1, W2, x)
        total += 0.5 * (d - Y2[0]) ** 2
    return total / len(samples)


def train(samples, targets, W1, W2, eta, alpha, epsilon, max_epochs):
    """Backpropagation com momentum:
           W(t+1) = W(t) + alpha * (W(t) - W(t-1)) + eta * delta * entrada
       Para quando |EQM_atual - EQM_anterior| <= epsilon.
    """
    W1 = W1.copy()
    W2 = W2.copy()
    W1_previous = W1.copy()
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
            delta2 = (d - Y2) * logistic_derivative(Y2)                   # saida
            delta1 = (delta2 @ W2[:, 1:]) * logistic_derivative(Y1[1:])    # escondida

            # Ajuste com momentum
            new_W2 = W2 + alpha * (W2 - W2_previous) + eta * np.outer(delta2, Y1)
            new_W1 = W1 + alpha * (W1 - W1_previous) + eta * np.outer(delta1, x)
            W2_previous, W2 = W2, new_W2
            W1_previous, W1 = W1, new_W1

        epoch += 1
        current_mse = mean_squared_error(W1, W2, samples, targets)
        mse_history.append(current_mse)

        if abs(current_mse - previous_mse) <= epsilon:
            return W1, W2, epoch, mse_history, True

    return W1, W2, epoch, mse_history, False


# ----------------------------------------------------------------------
# 3) Execucao dos 9 treinamentos (3 redes x 3 treinamentos)
# ----------------------------------------------------------------------
def main():
    series = load_series(TRAINING_FILE)
    print(f"Serie de treinamento: {len(series)} valores carregados.\n")

    table = {f"T{i}": {"Treinamento": f"T{i}"} for i in range(1, len(SEEDS) + 1)}
    histories = []
    weights = {}

    for net_name, (p, n_hidden) in NETWORKS.items():
        samples, targets = build_patterns(series, p)
        net_key = net_name.replace(" ", "")              # "Rede1", "Rede2", "Rede3"

        for i, seed in enumerate(SEEDS, start=1):
            name = f"T{i}"
            rng = np.random.default_rng(seed)
            W1 = rng.random((n_hidden, p + 1))           # N1 x (p + 1)
            W2 = rng.random((1, n_hidden + 1))           # 1 x (N1 + 1)

            print(f"Treinando {net_name} (p={p}, N1={n_hidden}) - {name}...")
            W1, W2, epochs, mse_history, converged = train(
                samples, targets, W1, W2, LEARNING_RATE, MOMENTUM, PRECISION, MAX_EPOCHS)
            if not converged:
                print(f"ATENCAO: {net_name} {name} atingiu {MAX_EPOCHS} epocas sem convergir!")

            table[name][f"{net_name} EQM"] = mse_history[-1]
            table[name][f"{net_name} Epocas"] = epochs
            for epoch, mse in enumerate(mse_history, start=1):
                histories.append({"network": net_name, "training": name,
                                  "epoch": epoch, "mse": mse})
            weights[f"W1_{net_key}_{name}"] = W1
            weights[f"W2_{net_key}_{name}"] = W2

    # ---- Tabela da Questao 2 ----
    results = pd.DataFrame(list(table.values()))
    print()
    print(results.to_string(index=False, float_format="{:.6f}".format))

    # ---- Salva os resultados ----
    results.to_csv("training_results_tdnn.csv", index=False, float_format="%.6f")
    pd.DataFrame(histories).to_csv("mse_history_tdnn.csv", index=False)
    np.savez("tdnn_weights.npz", **weights)
    print("\nTabela da Questao 2 salva em training_results_tdnn.csv")
    print("Historico do EQM salvo em mse_history_tdnn.csv")
    print("Pesos finais salvos em tdnn_weights.npz")


if __name__ == "__main__":
    main()
