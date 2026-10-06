import numpy as np
import pandas as pd

RESULTS_FILE = "training_results_adaline.csv"   # gerado pelo treinamento
TEST_FILE = "adaline_teste.csv"                 # sinais a classificar
WEIGHT_COLUMNS = ["w0_final", "w1_final", "w2_final", "w3_final", "w4_final"]


# ----------------------------------------------------------------------
# 1) Leitura dos sinais de teste
# ----------------------------------------------------------------------
def load_test_data(path):
    """Le o CSV de teste com pandas (colunas: Amostra, x1, x2, x3, x4)."""
    return pd.read_csv(path)


# ----------------------------------------------------------------------
# 2) Classificacao (agora SIM passamos u pela funcao de ativacao)
# ----------------------------------------------------------------------
def classify(w, x1, x2, x3, x4):
    """Calcula u = w0*(-1) + w1*x1 + w2*x2 + w3*x3 + w4*x4
       e aplica o degrau bipolar: +1 (valvula B) se u >= 0, senao -1 (valvula A).
    """
    x = np.array([-1.0, x1, x2, x3, x4])        # x0 = -1 (entrada do limiar)
    u = np.dot(w, x)
    return 1 if u >= 0 else -1


def valve_name(outputs):
    """Traduz as saidas dos 5 treinamentos para a valvula indicada ao comutador."""
    if all(y == -1 for y in outputs):
        return "A"
    if all(y == 1 for y in outputs):
        return "B"
    return "Divergente"                          # treinamentos discordaram


# ----------------------------------------------------------------------
# 3) Aplica os pesos finais de cada treinamento em todos os sinais
# ----------------------------------------------------------------------
def main():
    trainings = pd.read_csv(RESULTS_FILE)
    test_data = load_test_data(TEST_FILE)
    output = test_data.copy()

    y_columns = []
    for _, row in trainings.iterrows():
        w = row[WEIGHT_COLUMNS].to_numpy(dtype=float)
        column = f"y ({row['training']})"
        output[column] = [classify(w, s.x1, s.x2, s.x3, s.x4)
                          for s in test_data.itertuples()]
        y_columns.append(column)

    output["Valvula"] = output[y_columns].apply(lambda r: valve_name(r.tolist()), axis=1)

    print("Saidas: -1 = valvula A  |  +1 = valvula B\n")
    print(output.to_string(index=False))

    output.to_csv("classification_results_adaline.csv", index=False)
    print("\nTabela salva em classification_results_adaline.csv")


if __name__ == "__main__":
    main()