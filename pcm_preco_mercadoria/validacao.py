import numpy as np
import pandas as pd

TRAINING_FILE = "treinamento.csv"
TEST_FILE = "teste.csv"
WEIGHTS_FILE = "tdnn_weights.npz"
VALUE_COLUMN = "f(t)"
NETWORKS = {"Rede 1": 5, "Rede 2": 10, "Rede 3": 15}   # nome: p (numero de entradas)
TRAININGS = ["T1", "T2", "T3"]


# ----------------------------------------------------------------------
# 1) Leitura das series
# ----------------------------------------------------------------------
def load_series(path):
    """Le o CSV e devolve (instantes t, valores f(t))."""
    df = pd.read_csv(path)
    column = VALUE_COLUMN if VALUE_COLUMN in df.columns else df.columns[-1]
    return df[df.columns[0]].to_numpy(), df[column].to_numpy(dtype=float)


# ----------------------------------------------------------------------
# 2) Funcoes da rede (apenas a fase forward)
# ----------------------------------------------------------------------
def logistic(u):
    """Funcao de ativacao logistica: g(u) = 1 / (1 + e^-u)"""
    return 1.0 / (1.0 + np.exp(-u))


def predict(W1, W2, window):
    """Estima f(t) a partir da janela [x(t-1), ..., x(t-p)]."""
    x = np.concatenate(([-1.0], window))      # x0 = -1 (bias)
    Y1 = logistic(W1 @ x)
    Y1 = np.concatenate(([-1.0], Y1))
    return logistic(W2 @ Y1)[0]


# ----------------------------------------------------------------------
# 3) Validacao
# ----------------------------------------------------------------------
def main():
    _, train_values = load_series(TRAINING_FILE)
    test_t, test_values = load_series(TEST_FILE)
    full_series = np.concatenate((train_values, test_values))   # f(1) ... f(120)
    first_test = len(train_values)                               # indice de t = 101
    weights = np.load(WEIGHTS_FILE)

    output = pd.DataFrame({"Amostra": [f"t = {t}" for t in test_t], "f(t)": test_values})
    summary = []

    for net_name, p in NETWORKS.items():
        net_key = net_name.replace(" ", "")
        for name in TRAININGS:
            W1 = weights[f"W1_{net_key}_{name}"]
            W2 = weights[f"W2_{net_key}_{name}"]

            estimates = []
            for k in range(len(test_values)):
                t_index = first_test + k                          # posicao de f(t) na serie
                window = full_series[t_index - p:t_index][::-1]   # x(t-1), ..., x(t-p)
                estimates.append(predict(W1, W2, window))
            estimates = np.array(estimates)
            output[f"{net_name} ({name})"] = estimates

            # Erro relativo de cada amostra (%): |d - y| / |d| * 100
            relative_errors = np.abs(test_values - estimates) / np.abs(test_values) * 100
            summary.append({
                "network": net_name, "training": name,
                "mean_relative_error_pct": relative_errors.mean(),
                "variance_pct": relative_errors.var(ddof=0),
            })

    summary = pd.DataFrame(summary)

    print("Valores desejados e estimados (t = 101..120):\n")
    print(output.round(4).to_string(index=False))
    print("\nErro relativo medio e variancia:\n")
    print(summary.round(4).to_string(index=False))

    output.round(4).to_csv("validation_results_tdnn.csv", index=False)
    summary.round(4).to_csv("validation_summary_tdnn.csv", index=False)
    print("\nTabelas salvas em validation_results_tdnn.csv e validation_summary_tdnn.csv")


if __name__ == "__main__":
    main()
