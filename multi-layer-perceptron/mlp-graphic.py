import matplotlib.pyplot as plt
import pandas as pd

RESULTS_FILE = "training_results_mlp.csv"
HISTORY_FILE = "mse_history_mlp.csv"
OUTPUT_FILE = "graficos_eqm_mlp.png"
A4_SIZE_INCHES = (8.27, 11.69)        # folha A4 em pe


def main():
    results = pd.read_csv(RESULTS_FILE)
    history = pd.read_csv(HISTORY_FILE)

    # Os dois treinamentos com maior numero de epocas
    top_two = results.nlargest(2, "epochs")["training"].tolist()
    print(f"Treinamentos com mais epocas: {top_two}")

    # Uma folha com 2 graficos separados (nao superpostos), um embaixo do outro
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=A4_SIZE_INCHES)

    for ax, training in zip(axes, top_two):
        data = history[history["training"] == training]
        epochs = data["epoch"]
        mse = data["mse"]

        ax.plot(epochs, mse, color="tab:blue", linewidth=1.5)
        ax.set_title(f"Treinamento {training} - EQM em função das épocas")
        ax.set_xlabel("Épocas")
        ax.set_ylabel("Erro Quadrático Médio (EQM)")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Destaca o ponto final (onde o treinamento parou)
        last_epoch = epochs.iloc[-1]
        last_mse = mse.iloc[-1]
        ax.scatter([last_epoch], [last_mse], color="tab:red", zorder=3)
        ax.annotate(f"Época {last_epoch}\nEQM = {last_mse:.6f}",
                    xy=(last_epoch, last_mse),
                    xytext=(-130, 60), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="tab:red"))

    fig.suptitle("MLP - Backpropagation (η = 0.1, ε = 10⁻⁶)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(OUTPUT_FILE, dpi=150)
    print(f"Graficos salvos em {OUTPUT_FILE}")


if __name__ == "__main__":
    main()