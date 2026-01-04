import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np
from data_load import prepare_datasets

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

def main():
    print("--- Wczytywanie danych ---")

    (X_train_df, _), (X_test_df, y_test_series) = prepare_datasets()

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
    model.eval() # wyłącza droput i inne mechanizmy losowe
    with torch.no_grad(): # nie trzeba tego pamiętać do nauki gradientów
        train_reconstructions = model(train_tensor) # sprawdzenie jak wygląda norma - stworzenie "mapy normalności"


    # Ewaluacja na zbiorze testowym
    print("\n--- Testowanie ---")
    with torch.no_grad():
        test_reconstructions = model(test_tensor)

    # Skaner progów (Trade-off Analysis)
    print("\n--- WYNIKI ---")
    print(f"{'Percentyl':<8} | {'Próg':<8} | {'Recall':<8} | {'Precision':<8} | {'F1-Score':<8} | {'FP (Fałszywe alarmy)':<8} | {'TP (Wykryte ataki)':<8}")
    print("-" * 85)

    # Sprawdzamy zakres od 60. do 99. percentyla
    possible_percentiles = [60, 65, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99]
    
    # Obliczenie błędów
    with torch.no_grad():
        # Dla każdego wiersza (pakietu) bierze jego cechy, sumuje ich błędy i wyciąga średnią
        train_errors = torch.mean((train_tensor - train_reconstructions) ** 2, dim=1).numpy()
        test_errors = torch.mean((test_tensor - test_reconstructions) ** 2, dim=1).numpy()
        y_true = y_test_series.values.astype(int)

    best_f1 = 0
    best_p = 0

    for p in possible_percentiles:
        # Ustal próg na podstawie treningu (normy)
        threshold_temp = np.percentile(train_errors, p)
        
        # Zrób predykcję na teście
        y_pred_temp = (test_errors > threshold_temp).astype(int)
        
        # Policz statystyki
        tp = ((y_pred_temp == 1) & (y_true == 1)).sum()
        fp = ((y_pred_temp == 1) & (y_true == 0)).sum()
        fn = ((y_pred_temp == 0) & (y_true == 1)).sum()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{p:<10} | {threshold_temp:.6f} | {recall*100:6.2f}% | {precision*100:6.2f}% | {f1:.4f} | {fp} | {tp}")

        if f1 > best_f1:
            best_f1 = f1
            best_p = p

    print("-" * 85)
    print(f"Matematycznie najlepszy percentyl (wg F1-Score): {best_p}")

if __name__ == "__main__":
    main()