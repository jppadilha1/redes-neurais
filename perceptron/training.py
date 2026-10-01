# Na etapa de treinamento estamos fazendo um loop em todos os cenários do dataset
# É até a fórmula que multiplica entrada, peso, limiar. Bater o resultado de y com o d do dataset. Em todos os casos
# Geramos uma seed para cada treinamento e fazemos 5 treinamentos para obter 5 pesos ajustados para acertar tudo

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Configuracoes do exercicio
# ----------------------------------------------------------------------
TRAINING_FILE = "oleo_dataset.csv"
LEARNING_RATE = 0.01          # eta (dado pelo enunciado)
SEEDS = [1, 2, 3, 4, 5]       # uma semente por treinamento (T1 a T5)
MAX_EPOCHS = 10000            # limite de seguranca para nao travar


# ----------------------------------------------------------------------
# 1) Leitura do conjunto de treinamento
# ----------------------------------------------------------------------
def load_data(path):
    """Le o CSV com pandas e devolve:
       - samples: matriz (30 x 4) com as colunas [x0, x1, x2, x3], x0 = -1
       - targets: vetor com a classe correta d de cada amostra (-1 ou +1)
    """
    df = pd.read_csv(path)
    df.insert(0, "x0", -1.0)                  # entrada fixa do limiar
    samples = df[["x0", "x1", "x2", "x3"]].to_numpy()
    targets = df["d"].to_numpy()
    return samples, targets


# ----------------------------------------------------------------------
# 2) Funcoes do perceptron
# ----------------------------------------------------------------------
def activation(u):
    """Funcao degrau bipolar g(u): +1 se u >= 0, senao -1"""
    return 1.0 if u >= 0 else -1.0


def train(samples, targets, initial_weights, eta, max_epochs):
    """Treina o perceptron pela regra de Hebb.
       Repete epocas (passadas completas pelos dados) ate que uma
       epoca inteira termine sem nenhum erro.
       Devolve (final_weights, epochs, converged).
    """
    w = initial_weights.copy()               # copia para nao alterar o original
    epoch = 0

    while epoch < max_epochs:
        epoch += 1
        has_error = False

        for x, d in zip(samples, targets):
            u = np.dot(w, x)                 # u = w0*x0 + w1*x1 + w2*x2 + w3*x3
            y = activation(u)
            if y != d:
                # Regra de Hebb:  w <- w + eta * (d - y) * x
                w = w + eta * (d - y) * x
                has_error = True

        if not has_error:                    # epoca sem erros -> terminou
            return w, epoch, True

    return w, epoch, False                   # atingiu o limite sem convergir


# ----------------------------------------------------------------------
# 3) Execucao dos 5 treinamentos
# ----------------------------------------------------------------------
def main():
    samples, targets = load_data(TRAINING_FILE)
    print(f"Conjunto de treinamento: {len(samples)} amostras carregadas.\n")

    rows = []
    for i, seed in enumerate(SEEDS, start=1):
        # Reinicia o gerador com a semente deste treinamento
        rng = np.random.default_rng(seed)
        initial_weights = rng.random(4)      # 4 valores em [0, 1)

        final_weights, epochs, converged = train(
            samples, targets, initial_weights, LEARNING_RATE, MAX_EPOCHS)

        if not converged:
            print(f"ATENCAO: T{i} atingiu {MAX_EPOCHS} epocas sem convergir!")

        rows.append({
            "training": f"T{i}",
            "seed": seed,
            "w0_init": initial_weights[0], "w1_init": initial_weights[1],
            "w2_init": initial_weights[2], "w3_init": initial_weights[3],
            "w0_final": final_weights[0], "w1_final": final_weights[1],
            "w2_final": final_weights[2], "w3_final": final_weights[3],
            "epochs": epochs,
        })

    # ---- Tabela da Questao 2 ----
    results = pd.DataFrame(rows).round(4)
    print(results.to_string(index=False))

    # ---- Salva a tabela completa (os pesos finais serao usados na Questao 3) ----
    results.to_csv("training_results.csv", index=False)
    print("\nTabela completa salva em training_results.csv")


main()