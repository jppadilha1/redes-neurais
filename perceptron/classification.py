# Após o treinamento, aplicamos nossos pesos ajustados na tabela de testes

import numpy as np
import pandas as pd

RESULTS_FILE = "training_results.csv"   # gerado pelo treinamento
TEST_FILE = "oleo_teste.csv"            # amostras a classificar

# ----------------------------------------------------------------------
# 1) Leitura das amostras de teste
# ----------------------------------------------------------------------
def load_test_data(path):
    """Le o CSV de teste com pandas (colunas: Amostra, x1, x2, x3)."""
    return pd.read_csv(path)


# ----------------------------------------------------------------------
# 2) Funcao de classificacao (mesma do treinamento, mas sem ajustar pesos)
# ----------------------------------------------------------------------
def classify(w, x1, x2, x3):
    """Calcula u = w0*(-1) + w1*x1 + w2*x2 + w3*x3 e aplica o degrau bipolar."""
    x = np.array([-1.0, x1, x2, x3])     # x0 = -1 (entrada do limiar)
    u = np.dot(w, x)
    return 1 if u >= 0 else -1


# ----------------------------------------------------------------------
# 3) Aplica os pesos finais de cada treinamento em todas as amostras
# ----------------------------------------------------------------------
def main():
    trainings = pd.read_csv(RESULTS_FILE)
    test_data = load_test_data(TEST_FILE)
    output = test_data.copy()

    for _, row in trainings.iterrows():
        w = row[["w0_final", "w1_final", "w2_final", "w3_final"]].to_numpy(dtype=float)
        column = f"y ({row['training']})"
        output[column] = [classify(w, s.x1, s.x2, s.x3) for s in test_data.itertuples()]

    print("Saidas: -1 = classe C1  |  +1 = classe C2\n")
    print(output.to_string(index=False))

    output.to_csv("classification_results.csv", index=False)
    print("\nTabela salva em classification_results.csv")


main()