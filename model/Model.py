import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
import numpy as np
from collections import defaultdict
from data_load import prepare_datasets, load_dataset1, load_dataset3
from graph_draw import plot_tradeoff_analysis, plot_roc_curve, plot_error_histogram

class NetworkAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2), # funkcja aktywacji
            # RobustScaler produkuje wartości dodatnie (powyżej mediany) i ujemne (poniżej mediany)
            # LeakyReLu: jeśli liczba dodatnia - zostaw bez zmian, jeśli ujemna - przemnóż przez to co w nawiasie
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            
            nn.Linear(32, 8),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.LeakyReLU(0.2),
            
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


import pandas as pd
import numpy as np

def save_results_to_csv(results_accumulator, thresholds_map, filename="results/wyniki_analizy.csv"):
    print(f"\nZapisywanie wyników do pliku {filename}...")
    
    data_rows = []
    
    sorted_percentiles = sorted(results_accumulator.keys())

    for p in sorted_percentiles:
        stats = results_accumulator[p]
        threshold = thresholds_map[p]
        
        avg_recall = np.mean(stats['recall'])
        avg_precision = np.mean(stats['precision'])
        avg_f1 = np.mean(stats['f1'])
        std_f1 = np.std(stats['f1'])
        avg_fp = np.mean(stats['fp'])
        avg_tp = np.mean(stats['tp'])

        row = {
            "Percentyl": p,
            "Próg (Threshold)": threshold,
            "Avg Recall": avg_recall,
            "Avg Precision": avg_precision,
            "Avg F1": avg_f1,
            "F1 Std (+/-)": std_f1,
            "Avg FP": avg_fp,
            "Avg TP": avg_tp
        }
        data_rows.append(row)

    df = pd.DataFrame(data_rows)
    
    df.to_csv(filename, index=False, float_format='%.6f')

def main():
    print("--- Wczytywanie danych ---")

    (X_train_df, _), (X_test_df, y_test_series) = load_dataset3()

    print(f"Dane treningowe (Norma): {X_train_df.shape}")
    print(f"Dane testowe (Mix): {X_test_df.shape}")

    # Skalowanie danych
    X_train_values = X_train_df.values
    X_test_values = X_test_df.values

    scaler = RobustScaler() # skalowanie wokół mediany
    X_train_scaled = scaler.fit_transform(X_train_values)
    X_test_scaled = scaler.transform(X_test_values)
    
    # Konwersja na Tensory PyTorch
    train_tensor = torch.FloatTensor(X_train_scaled)
    test_tensor = torch.FloatTensor(X_test_scaled)

    # DataLoader
    train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=128, shuffle=True)

    # Inicjalizacja modelu
    input_dim = X_train_df.shape[1]
    model = NetworkAutoencoder(input_dim)
    criterion = nn.MSELoss() # Funkcja straty Mean Squared Error Loss (Błąd Średniokwadratowy) - kara za duże błędy
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # optymalizator - minimalizacja obliczonego wyżej błędu

    # Trening 
    print("\n--- Trening modelu ---")
    epochs = 10
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad() # zerowanie gradientów (czyszczenie)
            output = model(data) # odpowiedź modelu
            loss = criterion(output, target) # wylicz błąd
            loss.backward() # propagacja wsteczna - analiza błędu
            optimizer.step() # aktualizacja wiedzy
            train_loss += loss.item() # sumowanie błędów z tej epoki
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.6f}")


    # Ustalanie progu (Threshold) na zbiorze treningowym
    print("--- Testowanie ---")
    model.eval() # wyłącza droput i inne mechanizmy losowe
    
    with torch.no_grad(): # nie trzeba tego pamiętać do nauki gradientów
        train_reconstructions = model(train_tensor) # sprawdzenie jak wygląda norma - stworzenie "mapy normalności"
        
        # Ewaluacja na zbiorze testowym (Obliczamy rekonstrukcje raz dla całości)
        test_reconstructions = model(test_tensor)

    print("--- WYNIKI (Średnia z 10 losowych podzbiorów testowych) ---")
    
    # Sprawdzamy zakres od 60. do 99. percentyla
    possible_percentiles = [60, 65, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99]
    
    # Obliczenie błędów dla wszystkich danych (bazowe)
    with torch.no_grad():
        # Dla każdego wiersza (pakietu) bierze jego cechy, sumuje ich błędy i wyciąga średnią
        train_errors = torch.mean((train_tensor - train_reconstructions) ** 2, dim=1).numpy()
        test_errors_full = torch.mean((test_tensor - test_reconstructions) ** 2, dim=1).numpy()
        y_true_full = y_test_series.values.astype(int)

    # Wyliczamy progi raz, bo są stałe dla wytrenowanego modelu
    thresholds_map = {p: np.percentile(train_errors, p) for p in possible_percentiles}

    # Przygotowanie do pętli testowej
    NUM_RUNS = 10
    SAMPLE_FRACTION = 0.5 # Bierzemy 50% danych testowych w każdym przebiegu
    results_accumulator = defaultdict(lambda: defaultdict(list))
    
    total_samples = len(test_errors_full)
    subset_size = int(total_samples * SAMPLE_FRACTION)
    
    print(f"Przeprowadzam {NUM_RUNS} testów. W każdym losuję {subset_size} próbek z puli testowej...")

    # 3. Pętla testowa
    for run in range(NUM_RUNS):
        # Losujemy indeksy
        indices = np.random.choice(total_samples, subset_size, replace=False)
        
        # Wycinamy podzbiór błędów i etykiet
        test_errors_subset = test_errors_full[indices]
        y_true_subset = y_true_full[indices]
        
        # Sprawdzamy każdy percentyl na tym podzbiorze
        for p in possible_percentiles:
            threshold = thresholds_map[p]
            
            # Zrób predykcję na podzbiorze
            y_pred_temp = (test_errors_subset > threshold).astype(int)
            
            # Policz statystyki
            tp = ((y_pred_temp == 1) & (y_true_subset == 1)).sum()
            fp = ((y_pred_temp == 1) & (y_true_subset == 0)).sum()
            fn = ((y_pred_temp == 0) & (y_true_subset == 1)).sum()
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            
            # Zbierz wyniki
            results_accumulator[p]['recall'].append(recall)
            results_accumulator[p]['precision'].append(precision)
            results_accumulator[p]['f1'].append(f1)
            results_accumulator[p]['fp'].append(fp)
            results_accumulator[p]['tp'].append(tp)
        
        print(".", end="", flush=True)

    print("\n\n" + "-" * 105)
    print(f"{'Percentyl':<8} | {'Próg':<8} | {'Avg Recall':<12} | {'Avg Precision':<15} | {'Avg F1 (+/- std)':<18} | {'Avg FP':<10}")
    print("-" * 105)

    best_f1_avg = 0
    best_p = 0

    # Wyświetlanie uśrednionych wyników
    for p in possible_percentiles:
        stats = results_accumulator[p]
        threshold = thresholds_map[p]
        
        avg_recall = np.mean(stats['recall'])
        avg_precision = np.mean(stats['precision'])
        avg_f1 = np.mean(stats['f1'])
        std_f1 = np.std(stats['f1'])
        avg_fp = np.mean(stats['fp'])
        
        f1_str = f"{avg_f1:.4f} (+/-{std_f1:.3f})"
        
        print(f"{p:<10} | {threshold:.6f} | {avg_recall*100:6.2f}%      | {avg_precision*100:6.2f}%         | {f1_str:<18} | {avg_fp:<10.1f}")

        if avg_f1 > best_f1_avg:
            best_f1_avg = avg_f1
            best_p = p

    print("-" * 105)
    print(f"Matematycznie najlepszy percentyl (wg średniego F1-Score): {best_p}")

    # Wykres recall vs precision
    plot_tradeoff_analysis(results_accumulator, possible_percentiles)

    # Wykres ROC
    plot_roc_curve(y_true_full, test_errors_full)

    # Histogram
    best_threshold = thresholds_map[best_p]
    plot_error_histogram(y_true_full, test_errors_full, threshold=best_threshold)

    # Zapis wyników do pliku
    save_results_to_csv(results_accumulator, thresholds_map)

if __name__ == "__main__":
    main()