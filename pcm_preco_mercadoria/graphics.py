import re

import matplotlib.pyplot as plt
import pandas as pd

HISTORY_FILE = "mse_history_tdnn.csv"
RESULTS_FILE = "validation_results_tdnn.csv"
SUMMARY_FILE = "validation_summary_tdnn.csv"
NETWORK_INFO = {"Rede 1": "p = 5, N1 = 10", "Rede 2": "p = 10, N1 = 15", "Rede 3": "p = 15, N1 = 25"}
A4_SIZE_INCHES = (8.27, 11.69)        # folha A4 em pe


def best_trainings(summary):
    """Devolve {rede: melhor treinamento}, pelo menor erro relativo medio."""
    best = summary.loc[summary.groupby("network")["mean_relative_error_pct"].idxmin()]
    return dict(zip(best["network"], best["training"])), best


def plot_mse(history, best):
    """Questao 4: EQM x epocas do melhor treinamento de cada rede."""
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=A4_SIZE_INCHES)
    for ax, (network, training) in zip(axes, best.items()):
        data = history[(history["network"] == network) & (history["training"] == training)]
        epochs, mse = data["epoch"], data["mse"]

        ax.plot(epochs, mse, color="tab:blue", linewidth=1.5)
        ax.set_title(f"{network} ({NETWORK_INFO[network]}) - treinamento {training}")
        ax.set_xlabel("Épocas")
        ax.set_ylabel("EQM")
        ax.grid(True, linestyle="--", alpha=0.5)

        last_epoch, last_mse = epochs.iloc[-1], mse.iloc[-1]
        ax.scatter([last_epoch], [last_mse], color="tab:red", zorder=3)
        ax.annotate(f"Época {last_epoch}\nEQM = {last_mse:.6f}",
                    xy=(last_epoch, last_mse), xytext=(-130, 45),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="tab:red"))

    fig.suptitle("TDNN - EQM em função das épocas (melhor treinamento de cada rede)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("graficos_eqm_tdnn.png", dpi=150)
    plt.close(fig)


def plot_forecast(results, best, best_rows):
    """Questao 5: valores desejados x estimados para t = 101..120."""
    t = [int(re.findall(r"\d+", str(v))[-1]) for v in results["Amostra"]]
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=A4_SIZE_INCHES)
    for ax, (network, training) in zip(axes, best.items()):
        error = best_rows.loc[best_rows["network"] == network, "mean_relative_error_pct"].iloc[0]

        ax.plot(t, results["f(t)"], marker="o", color="tab:blue", linewidth=1.5,
                label="Desejado f(t)")
        ax.plot(t, results[f"{network} ({training})"], marker="s", linestyle="--",
                color="tab:orange", linewidth=1.5, label=f"Estimado pela rede ({training})")
        ax.set_title(f"{network} ({NETWORK_INFO[network]}) - {training} "
                     f"- erro relativo médio: {error:.2f}%")
        ax.set_xlabel("t")
        ax.set_ylabel("f(t)")
        ax.set_xticks(t)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(top=0.95)
        ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("TDNN - Valores desejados x estimados (t = 101..120)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("graficos_previsao_tdnn.png", dpi=150)
    plt.close(fig)


def main():
    history = pd.read_csv(HISTORY_FILE)
    results = pd.read_csv(RESULTS_FILE)
    summary = pd.read_csv(SUMMARY_FILE)

    best, best_rows = best_trainings(summary)
    print("Melhor treinamento de cada rede (menor erro relativo medio):")
    for _, row in best_rows.iterrows():
        print(f"  {row['network']}: {row['training']} "
              f"({row['mean_relative_error_pct']:.2f}%)")

    plot_mse(history, best)
    plot_forecast(results, best, best_rows)
    print("\nGraficos salvos em graficos_eqm_tdnn.png e graficos_previsao_tdnn.png")


if __name__ == "__main__":
    main()
