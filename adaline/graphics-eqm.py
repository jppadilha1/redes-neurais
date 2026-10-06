import matplotlib.pyplot as plt
import pandas as pd

HISTORY_FILE = "mse_history.csv"
TRAININGS_TO_PLOT = ["T1", "T2"]     # os dois primeiros treinamentos
A4_SIZE_INCHES = (8.27, 11.69)        # folha A4 em pe


def main():
    history = pd.read_csv(HISTORY_FILE)

    # Uma figura (folha) com 2 graficos, um embaixo do outro
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=A4_SIZE_INCHES)

    for ax, training in zip(axes, TRAININGS_TO_PLOT):
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
        ax.annotate(f"Época {last_epoch}\nEQM = {last_mse:.4f}",
                    xy=(last_epoch, last_mse),
                    xytext=(-110, 40), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="tab:red"))

    fig.suptitle("ADALINE - Regra Delta (η = 0.0025, ε = 10⁻⁶)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig("graficos_eqm.pdf")
    fig.savefig("graficos_eqm.png", dpi=150)
    print("Graficos salvos em graficos_eqm.pdf e graficos_eqm.png")
    plt.show()


if __name__ == "__main__":
    main()