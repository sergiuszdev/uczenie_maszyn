import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import RobustScaler
from pathlib import Path

import pandas as pd


def save_results_to_csv(results_accumulator, thresholds_map, file_tag=""):
    filename = f"results/results_{file_tag}.csv"
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    print(f"\nZapisywanie wyników do pliku {filename}...")

    data_rows = []

    sorted_percentiles = sorted(results_accumulator.keys())

    for p in sorted_percentiles:
        stats = results_accumulator[p]
        threshold = thresholds_map[p]

        avg_recall = np.mean(stats["recall"])
        avg_precision = np.mean(stats["precision"])
        avg_f1 = np.mean(stats["f1"])
        std_f1 = np.std(stats["f1"])
        avg_fp = np.mean(stats["fp"])
        avg_tp = np.mean(stats["tp"])

        row = {
            "Percentyl": p,
            "Próg (Threshold)": threshold,
            "Avg Recall": avg_recall,
            "Avg Precision": avg_precision,
            "Avg F1": avg_f1,
            "F1 Std (+/-)": std_f1,
            "Avg FP": avg_fp,
            "Avg TP": avg_tp,
        }
        data_rows.append(row)

    df = pd.DataFrame(data_rows)

    df.to_csv(filename, index=False, float_format="%.6f")


def plot_tradeoff_analysis(results_accumulator, percentiles, file_tag=""):
    filename = f"results/analiza_percentyli_{file_tag}.png"
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    sorted_percentiles = sorted(percentiles)

    y_recall_avg = []
    y_precision_avg = []
    y_f1_avg = []

    for p in sorted_percentiles:
        stats = results_accumulator[p]
        y_recall_avg.append(np.mean(stats["recall"]))
        y_precision_avg.append(np.mean(stats["precision"]))
        y_f1_avg.append(np.mean(stats["f1"]))

    plt.figure(figsize=(10, 6))

    plt.plot(
        sorted_percentiles,
        y_recall_avg,
        label="Średni Recall (Czułość)",
        color="blue",
        marker="o",
        linestyle="-",
        linewidth=2,
    )

    plt.plot(
        sorted_percentiles,
        y_precision_avg,
        label="Średnia Precision (Precyzja)",
        color="green",
        marker="s",
        linestyle="-",
        linewidth=2,
    )

    plt.plot(
        sorted_percentiles,
        y_f1_avg,
        label="Średni F1-Score",
        color="red",
        marker="^",
        linestyle="--",
        alpha=0.7,
    )

    plt.title(
        "Zmiana wartości parametrów Recall i Precision w zależności od ustalonego percentyla",
        fontsize=14,
    )
    plt.xlabel("Percentyl (Próg odcięcia)", fontsize=12)
    plt.ylabel("Wartość współczynnika (0.0 - 1.0)", fontsize=12)

    plt.xticks(sorted_percentiles)
    plt.yticks(np.arange(0, 1.1, 0.1))

    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="best")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Wykres został zapisany jako '{filename}'")


def plot_roc_curve(y_true, y_scores, file_tag=""):
    filename = f"results/roc_curve_{file_tag}.png"
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"Krzywa ROC (AUC = {roc_auc:.4f})"
    )

    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (Odsetek Fałszywych Alarmów)", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Wykres został zapisany jako '{filename}'")


def plot_error_histogram(y_true, errors, threshold=None, file_tag=""):
    filename = f"results/error_histogram_{file_tag}.png"
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    errors_normal = errors[y_true == 0]
    errors_attack = errors[y_true == 1]

    plt.figure(figsize=(12, 6))

    sns.histplot(
        errors_normal,
        color="green",
        label="Ruch normalny",
        kde=False,
        bins=100,
        alpha=0.3,
        element="step",
    )
    sns.histplot(
        errors_attack,
        color="red",
        label="Atak",
        kde=False,
        bins=100,
        alpha=0.3,
        element="step",
    )

    if threshold is not None:
        plt.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Próg ({threshold:.5f})",
        )

    plt.yscale("log")
    plt.ylim(bottom=1)

    plt.title("Rozkład błędu rekonstrukcji: norma vs atak", fontsize=14)
    plt.xlabel("Wartość błędu rekonstrukcji", fontsize=12)
    plt.ylabel("Liczba pakietów (skala logarytmiczna)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Wykres zapisany jako '{filename}'")


def plot_preprocessing_effect(df, feature_name, file_tag=""):
    filename = f"results/preprocessing_effect_{file_tag}.png"
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    if feature_name not in df.columns:
        print(
            f"Błąd: Nie znaleziono kolumny '{feature_name}' w danych. Wykres nie zostanie wygenerowany"
        )
        return

    raw_data = df[feature_name].dropna().values

    log_data = np.log1p(np.clip(raw_data, 0, None))

    scaler = RobustScaler()

    processed_data = scaler.fit_transform(log_data.reshape(-1, 1)).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.histplot(raw_data, bins=50, ax=axes[0], color="gray", kde=True, stat="density")
    axes[0].set_title(f'Przed: Surowe dane ("{feature_name}")', fontsize=12)
    axes[0].set_xlabel("Wartość oryginalna")
    axes[0].set_ylabel("Gęstość")
    axes[0].set_yscale("log")

    sns.histplot(
        processed_data, bins=50, ax=axes[1], color="blue", kde=True, stat="density"
    )
    axes[1].set_title(f'Po: Logarytm + RobustScaler ("{feature_name}")', fontsize=12)
    axes[1].set_xlabel("Wartość przeskalowana")
    axes[1].set_ylabel("Gęstość")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Wykres zapisany jako '{filename}'")


def plot_feature_separability(df, y_true, feature_name, apply_log=True, file_tag=""):
    filename = f"results/feature_separability_{file_tag}.png"
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    if feature_name not in df.columns:
        print(
            f"Błąd: Nie znaleziono kolumny '{feature_name}' w danych. Wykres nie zostanie wygenerowany"
        )
        return

    data = df[feature_name].values

    if apply_log:
        plot_data = np.log1p(np.clip(data, 0, None))
        xlabel_text = f'Logarytm z wartości "{feature_name}"'
    else:
        plot_data = data
        xlabel_text = f'Wartość "{feature_name}"'

    data_normal = plot_data[y_true == 0]
    data_attack = plot_data[y_true == 1]

    plt.figure(figsize=(10, 6))

    sns.kdeplot(data_normal, color="green", label="Ruch normalny", fill=True, alpha=0.3)
    sns.kdeplot(data_attack, color="red", label="Atak", fill=True, alpha=0.3)

    plt.title(f"Separowalność klas na podstawie cechy: {feature_name}", fontsize=14)
    plt.xlabel(xlabel_text, fontsize=12)
    plt.ylabel("Gęstość prawdopodobieństwa", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Wykres zapisany jako '{filename}'")
