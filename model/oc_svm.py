import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from collections import defaultdict

from data_logic.preprocess_datasets import (
    load_dataset2,
    load_dataset1,
    load_dataset3,
    get_random_seed,
)
from data_logic.process_results import (
    plot_tradeoff_analysis,
    plot_roc_curve,
    plot_error_histogram,
    plot_feature_separability,
    plot_preprocessing_effect,
    save_results_to_csv,
)


def make_tests_ocsvm(load_data: (), tag="", save_results=False):
    print("--- Wczytywanie danych ---")

    (X_train_df, _), (X_test_df, y_test_series) = load_data()
    tag_full = f"{tag}_ocsvm_sgd"
    if save_results:
        plot_preprocessing_effect(
            X_train_df, feature_name="IN_BYTES", file_tag=tag_full
        )
        plot_feature_separability(
            X_test_df,
            y_test_series.values,
            feature_name="IN_PKTS",
            apply_log=True,
            file_tag=tag_full,
        )

    print(f"Dane treningowe (Norma): {X_train_df.shape}")
    print(f"Dane testowe (Mix): {X_test_df.shape}")

    X_train_values = X_train_df.values
    X_test_values = X_test_df.values

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_values)
    X_test_scaled = scaler.transform(X_test_values)

    # nu -> upper bound on the fraction of training errors and a lower bound of the fraction of support vectors
    print("\n--- Trening modelu ---")

    model = make_pipeline(
        Nystroem(gamma=0.1, random_state=get_random_seed(), n_components=100),
        SGDOneClassSVM(nu=0.01, random_state=get_random_seed()),
    )

    # ten svm jest strasznie wolny
    # model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)

    model.fit(X_train_scaled)

    print("--- Testowanie ---")

    train_scores_raw = model.decision_function(X_train_scaled).flatten()
    train_errors = -1 * train_scores_raw

    test_scores_raw = model.decision_function(X_test_scaled).flatten()
    test_errors_full = -1 * test_scores_raw

    y_true_full = y_test_series.values.astype(int)

    print("--- WYNIKI (Średnia z 10 losowych podzbiorów testowych) ---")

    possible_percentiles = [60, 65, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99]

    thresholds_map = {p: np.percentile(train_errors, p) for p in possible_percentiles}

    NUM_RUNS = 10
    SAMPLE_FRACTION = 0.5
    results_accumulator = defaultdict(lambda: defaultdict(list))

    total_samples = len(test_errors_full)
    subset_size = int(total_samples * SAMPLE_FRACTION)

    print(
        f"Przeprowadzam {NUM_RUNS} testów. W każdym losuję {subset_size} próbek z puli testowej..."
    )

    for run in range(NUM_RUNS):
        indices = np.random.choice(total_samples, subset_size, replace=False)
        test_errors_subset = test_errors_full[indices]
        y_true_subset = y_true_full[indices]

        for p in possible_percentiles:
            threshold = thresholds_map[p]

            y_pred_temp = (test_errors_subset > threshold).astype(int)

            tp = ((y_pred_temp == 1) & (y_true_subset == 1)).sum()
            fp = ((y_pred_temp == 1) & (y_true_subset == 0)).sum()
            fn = ((y_pred_temp == 0) & (y_true_subset == 1)).sum()

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

            results_accumulator[p]["recall"].append(recall)
            results_accumulator[p]["precision"].append(precision)
            results_accumulator[p]["f1"].append(f1)
            results_accumulator[p]["fp"].append(fp)
            results_accumulator[p]["tp"].append(tp)

        print(".", end="", flush=True)

    print("\n\n" + "-" * 105)
    print(
        f"{'Percentyl':<8} | {'Próg':<8} | {'Avg Recall':<12} | {'Avg Precision':<15} | {'Avg F1 (+/- std)':<18} | {'Avg FP':<10}"
    )
    print("-" * 105)

    best_f1_avg = 0
    best_p = 0

    for p in possible_percentiles:
        stats = results_accumulator[p]
        threshold = thresholds_map[p]

        avg_recall = np.mean(stats["recall"])
        avg_precision = np.mean(stats["precision"])
        avg_f1 = np.mean(stats["f1"])
        std_f1 = np.std(stats["f1"])
        avg_fp = np.mean(stats["fp"])

        f1_str = f"{avg_f1:.4f} (+/-{std_f1:.3f})"

        print(
            f"{p:<10} | {threshold:.6f} | {avg_recall * 100:6.2f}%      | {avg_precision * 100:6.2f}%         | {f1_str:<18} | {avg_fp:<10.1f}"
        )

        if avg_f1 > best_f1_avg:
            best_f1_avg = avg_f1
            best_p = p

    print("-" * 105)
    print(f"Matematycznie najlepszy percentyl (wg średniego F1-Score): {best_p}")

    if save_results:
        plot_tradeoff_analysis(
            results_accumulator, possible_percentiles, file_tag=tag_full
        )
        plot_roc_curve(y_true_full, test_errors_full, file_tag=tag_full)
        best_threshold = thresholds_map[best_p]
        plot_error_histogram(
            y_true_full, test_errors_full, threshold=best_threshold, file_tag=tag_full
        )
        save_results_to_csv(results_accumulator, thresholds_map, file_tag=tag_full)


if __name__ == "__main__":
    make_tests_ocsvm(load_dataset1, tag="kdd99", save_results=True)
    make_tests_ocsvm(load_dataset2, tag="netflowv9", save_results=True)
    make_tests_ocsvm(load_dataset3, tag="coresiot", save_results=True)
