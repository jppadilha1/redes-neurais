import numpy as np
import pandas as pd

TEST_FILE = "conjunto_validacao.csv"
WEIGHTS_FILE = "mlp_weights.npz"
TRAININGS = ["T1", "T2", "T3", "T4", "T5"]


# ----------------------------------------------------------------------
# 1) Funcoes da rede (mesmas do treinamento, apenas a fase forward)
# ----------------------------------------------------------------------
def logistic(u):
    """Funcao de ativacao logistica: g(u) = 1 / (1 + e^-u)"""
    return 1.0 / (1.0 + np.exp(-u))


def predict(W1, W2, x1, x2, x3):
    """Fase forward: calcula a saida da rede para uma amostra (sem ajustar pesos)."""
    x = np.array([-1.0, x1, x2, x3])          # x0 = -1 (bias)
    Y1 = logistic(W1 @ x)                     # saida dos 10 neuronios escondidos
    Y1 = np.concatenate(([-1.0], Y1))         # bias para a camada de saida
    Y2 = logistic(W2 @ Y1)                    # saida da rede
    return Y2[0]


# ----------------------------------------------------------------------
# 2) Validacao
# ----------------------------------------------------------------------
def main():
    test_data = pd.read_csv(TEST_FILE)
    weights = np.load(WEIGHTS_FILE)
    output = test_data.copy()

    summary = []
    for name in TRAININGS:
        W1 = weights[f"W1_{name}"]
        W2 = weights[f"W2_{name}"]

        # Saida da rede para cada amostra de teste
        y_net = np.array([predict(W1, W2, s.x1, s.x2, s.x3)
                          for s in test_data.itertuples()])
        output[f"y ({name})"] = y_net

        # Erro relativo de cada amostra (%):  |d - y| / |d| * 100
        relative_errors = np.abs(test_data["d"] - y_net) / np.abs(test_data["d"]) * 100

        summary.append({
            "training": name,
            "mean_relative_error_pct": relative_errors.mean(),   # media dos 20 erros
            "variance_pct": relative_errors.var(ddof=0),          # variancia dos 20 erros
        })

    summary = pd.DataFrame(summary)

    # ---- Tabela da Questao 5 ----
    print("Saidas da rede para o conjunto de teste:\n")
    print(output.round(4).to_string(index=False))
    print("\nErro relativo medio e variancia por treinamento:\n")
    print(summary.round(4).to_string(index=False))

    best = summary.loc[summary["mean_relative_error_pct"].idxmin(), "training"]
    print(f"\nMenor erro relativo medio: {best}")

    # ---- Salva os resultados ----
    output.round(4).to_csv("validation_results_mlp.csv", index=False)
    summary.round(4).to_csv("validation_summary_mlp.csv", index=False)
    print("\nTabelas salvas em validation_results_mlp.csv e validation_summary_mlp.csv")


if __name__ == "__main__":
    main()