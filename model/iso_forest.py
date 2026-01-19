import numpy as np
from sklearn.ensemble import IsolationForest
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

from data_logic.preprocess_datasets import (
    load_dataset2,
    load_dataset1,
    load_dataset3,
    get_random_seed,
    load_dataset1_full,
    load_dataset2_full,
    load_dataset3_full
)
from data_logic.process_results import (
    plot_tradeoff_analysis,
    plot_roc_curve,
    plot_error_histogram,
    plot_feature_separability,
    plot_preprocessing_effect,
    save_results_to_csv,
)


def make_tests_isoforest(load_data, tag="", save_results=False):
    print("--- Wczytywanie danych ---")

    (X_train_df, _), (X_test_df, y_test_series) = load_data()
    tag_full = f"{tag}_isoforest"

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

    print("\n--- Trening modelu ---")

    model = IsolationForest(
        n_estimators=300,
        max_samples=256,
        contamination="auto",
        random_state=get_random_seed(),
        n_jobs=-1,
    )

    model.fit(X_train_values)

    print("--- Testowanie ---")

    train_scores_raw = model.decision_function(X_train_values)
    train_errors = -train_scores_raw

    test_scores_raw = model.decision_function(X_test_values)
    test_errors_full = -test_scores_raw

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


def run_isoforest_cross_validation(data_loader_func, tag="isoforest_cv", n_splits=5):

    X_all, y_all = data_loader_func()
    
    if len(X_all) == 0:
        print("Błąd: Pusty zbiór danych!")
        return

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results_accumulator = defaultdict(lambda: defaultdict(list))
    thresholds_accumulator = defaultdict(list)
    
    y_true_global = []
    test_scores_global = [] # Tutaj trzymamy "błędy" (odwrócone score)

    possible_percentiles = [60, 65, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99]

    # PĘTLA WALIDACJI KRZYŻOWEJ 
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
        print(f"\n=== FOLD {fold_i + 1}/{n_splits} ===")

        X_train_raw = X_all.iloc[train_idx]
        y_train_raw = y_all.iloc[train_idx]
        X_test_fold = X_all.iloc[test_idx]
        y_test_fold = y_all.iloc[test_idx]

        train_mask_normal = (y_train_raw == 0)
        X_train_normal = X_train_raw[train_mask_normal]

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_normal.values)
        X_test_scaled = scaler.transform(X_test_fold.values)

        model = IsolationForest(
            n_estimators=300,
            max_samples=256,
            contamination='auto',
            n_jobs=-1,
            random_state=get_random_seed()
        )
        
        model.fit(X_train_scaled)
        
        train_scores_raw = model.decision_function(X_train_scaled)
        train_errors = -train_scores_raw  # Teraz: mała wartość = norma, duża = atak
        
        test_scores_raw = model.decision_function(X_test_scaled)
        test_errors = -test_scores_raw

        y_true_global.extend(y_test_fold.values)
        test_scores_global.extend(test_errors)

        for p in possible_percentiles:
            threshold = np.percentile(train_errors, p)
            thresholds_accumulator[p].append(threshold)

            y_pred = (test_errors > threshold).astype(int)

            tp = ((y_pred == 1) & (y_test_fold == 1)).sum()
            fp = ((y_pred == 1) & (y_test_fold == 0)).sum()
            fn = ((y_pred == 0) & (y_test_fold == 1)).sum()

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            results_accumulator[p]['recall'].append(recall)
            results_accumulator[p]['precision'].append(precision)
            results_accumulator[p]['f1'].append(f1)
            results_accumulator[p]['fp'].append(fp)
            results_accumulator[p]['tp'].append(tp)

        print(f"   -> Fold {fold_i+1} zakończony. F1 (p95): {results_accumulator[95]['f1'][-1]:.4f}")

    print("\n" + "="*105)
    print(f"WYNIKI CV (Isolation Forest) - {tag}")
    print("="*105)
    print(f"{'Percentyl':<10} | {'Avg Thresh':<12} | {'Avg Recall':<12} | {'Avg Precision':<15} | {'Avg F1 (+/- std)':<20} | {'Avg FP':<10}")
    print("-" * 105)

    best_f1_avg = 0
    best_p = 0
    avg_thresholds_map = {}

    for p in possible_percentiles:
        avg_thresh = np.mean(thresholds_accumulator[p])
        avg_thresholds_map[p] = avg_thresh
        
        stats = results_accumulator[p]
        avg_rec = np.mean(stats['recall'])
        avg_prec = np.mean(stats['precision'])
        avg_f1 = np.mean(stats['f1'])
        std_f1 = np.std(stats['f1'])
        avg_fp = np.mean(stats['fp'])

        print(f"{p:<10} | {avg_thresh:.6f}     | {avg_rec*100:6.2f}%      | {avg_prec*100:6.2f}%         | {avg_f1:.4f} (+/- {std_f1:.3f}) | {avg_fp:<10.1f}")
        
        if avg_f1 > best_f1_avg:
            best_f1_avg = avg_f1
            best_p = p

    print("-" * 105)
    print(f"Najlepszy średni percentyl: {best_p}")

    y_true_global = np.array(y_true_global)
    test_scores_global = np.array(test_scores_global)
    plot_tradeoff_analysis(results_accumulator, possible_percentiles, file_tag=tag)
    plot_roc_curve(y_true_global, test_scores_global, file_tag=tag)
    
    avg_best_threshold = avg_thresholds_map[best_p]
    plot_error_histogram(
        y_true_global, 
        test_scores_global, 
        threshold=avg_best_threshold, 
        file_tag=tag
    )

    save_results_to_csv(results_accumulator, avg_thresholds_map, file_tag=tag)


if __name__ == "__main__":
    #make_tests_isoforest(load_dataset1, tag="kdd99", save_results=True)
    #make_tests_isoforest(load_dataset2, tag="netflowv9", save_results=True)
    #make_tests_isoforest(load_dataset3, tag="coresiot", save_results=True)

    #run_isoforest_cross_validation(load_dataset1_full, tag="kdd", n_splits=5)
    #run_isoforest_cross_validation(load_dataset2_full, tag="netflow", n_splits=5)
    run_isoforest_cross_validation(load_dataset3_full, tag="coresiot", n_splits=5)
