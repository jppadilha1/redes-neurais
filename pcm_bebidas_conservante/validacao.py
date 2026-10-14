import numpy as np
import pandas as pd

TEST_FILE = "teste.csv"
WEIGHTS_FILE = "mlp_conservante_weights.npz"
INPUT_COLUMNS = ["x1", "x2", "x3", "x4"]
TARGET_COLUMNS = ["d1", "d2", "d3"]
METHODS = {
    "standard": "Backpropagation padrao",
    "momentum": "Backpropagation com momentum",
}
CLASS_NAMES = {(1, 0, 0): "A", (0, 1, 0): "B", (0, 0, 1): "C"}


# ----------------------------------------------------------------------
# 1) Funcoes da rede (mesmas do treinamento, apenas a fase forward)
# ----------------------------------------------------------------------
def logistic(u):
    """Funcao de ativacao logistica: g(u) = 1 / (1 + e^-u)"""
    return 1.0 / (1.0 + np.exp(-u))


def predict(W1, W2, inputs):
    """Fase forward: devolve as 3 saidas reais da rede (sem ajustar pesos)."""
    x = np.concatenate(([-1.0], inputs))      # x0 = -1 (bias)
    Y1 = logistic(W1 @ x)                     # saida dos 15 neuronios escondidos
    Y1 = np.concatenate(([-1.0], Y1))         # bias para a camada de saida
    return logistic(W2 @ Y1)                  # y1, y2, y3


# ----------------------------------------------------------------------
# 2) Questao 3: pos-processamento (arredondamento simetrico)
# ----------------------------------------------------------------------
def post_process(y):
    """Converte as saidas reais em inteiros:  1 se y_i >= 0.5,  0 se y_i < 0.5"""
    return (y >= 0.5).astype(int)


def class_name(outputs):
    """Traduz (y1, y2, y3) pos-processado para o tipo de conservante."""
    return CLASS_NAMES.get(tuple(outputs), "Indefinido")   # ex.: (0,0,0) ou (1,1,0)


# ----------------------------------------------------------------------
# 3) Questao 4: validacao com o conjunto de teste
# ----------------------------------------------------------------------
def main():
    test_data = pd.read_csv(TEST_FILE)
    weights = np.load(WEIGHTS_FILE)
    inputs = test_data[INPUT_COLUMNS].to_numpy(dtype=float)
    targets = test_data[TARGET_COLUMNS].to_numpy(dtype=int)

    summary = []
    for key, label in METHODS.items():
        W1 = weights[f"W1_{key}"]
        W2 = weights[f"W2_{key}"]

        raw = np.array([predict(W1, W2, x) for x in inputs])      # saidas reais
        post = np.array([post_process(y) for y in raw])           # saidas 0/1

        # Uma amostra so conta como acerto se as 3 saidas baterem com d1, d2, d3
        hits = np.all(post == targets, axis=1)
        accuracy = hits.mean() * 100

        output = test_data.copy()
        for j in range(3):
            output[f"y{j+1}_pos"] = post[:, j]
        for j in range(3):
            output[f"y{j+1}"] = raw[:, j].round(4)
        output["Tipo esperado"] = [class_name(d) for d in targets]
        output["Tipo rede"] = [class_name(p) for p in post]
        output["Acerto"] = np.where(hits, "Sim", "Nao")

        print(f"===== {label} =====\n")
        print(output.to_string(index=False))
        print(f"\nTaxa de acerto: {hits.sum()}/{len(hits)} = {accuracy:.2f}%\n")

        output.to_csv(f"validation_results_conservante_{key}.csv", index=False)
        summary.append({"method": key, "hits": int(hits.sum()),
                        "total": len(hits), "accuracy_pct": round(accuracy, 2)})

    pd.DataFrame(summary).to_csv("validation_summary_conservante.csv", index=False)
    print("Resultados salvos em validation_results_conservante_<metodo>.csv")
    print("Resumo salvo em validation_summary_conservante.csv")


if __name__ == "__main__":
    main()
