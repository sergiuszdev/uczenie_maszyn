import pandas as pd
import torch
import numpy as np

from CONFIG import NETFLOW_V9_TRAIN

def parse_netflow(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df

def preprocess_netflow(df):
    # Wykluczone kolumny
    exclude_cols = [
        "ALERT", "ANOMALY", 
        "ID", "FLOW_ID", 
        "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
        "ANALYSIS_TIMESTAMP", "FIRST_SWITCHED", "LAST_SWITCHED",
        "L4_SRC_PORT", 
        "PROTOCOL_MAP", "TOTAL_FLOWS_EXP", "TOTAL_PKTS_EXP", "TOTAL_BYTES_EXP"
    ]

    # Kolumny do logarytmizacji
    log_cols = [
        "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", 
        "FLOW_DURATION_MILLISECONDS", 
        "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
        "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
        "TCP_WIN_MIN_IN", "TCP_WIN_MIN_OUT",
        "TCP_WIN_MSS_IN"
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].copy()

    # Logarytmizacja
    # - zmniejszenie gigantycznych wartości,
    # - uwidocznienie różnic między małymi wartościami
    for col in X.columns:
        if col in log_cols:
            X[col] = np.log1p(X[col])

    X = X.fillna(0)
    X = X.astype(float)
    return X, feature_cols

def prepare_datasets():
    """
    Wczytuje jeden plik treningowy i dzieli go na:
    1. Czysty zbiór treningowy (500k próbek Normy)
    2. Zbiór testowy (1M próbek Mixu)
    """
    df = parse_netflow(NETFLOW_V9_TRAIN)

    if "ALERT" in df.columns:
        df["ALERT"] = df["ALERT"].fillna("None")
    else:
        raise ValueError("Brak kolumny ALERT w pliku treningowym!")

    print(f"Rozmiar całkowity: {len(df)}")

    # Wybór danych do TRENINGU (Tylko Norma)
    df_normal = df[df["ALERT"] == "None"]
    
    train_size = 500000
    if len(df_normal) >= train_size:
        print(f"Losowanie {train_size} próbek normy do treningu...")
        df_train = df_normal.sample(n=train_size, random_state=42)
    else:
        print(f"UWAGA: Dostępne tylko {len(df_normal)} próbek normy.")
        df_train = df_normal.copy()

    #Wybór danych do TESTU (Mix: Ataki + pozostała Norma)
    # Usuwamy z głównego df te wiersze, które wzięliśmy do treningu
    train_indices = df_train.index
    df_remaining = df.drop(train_indices)
    
    test_size = 1000000
    if len(df_remaining) >= test_size:
        print(f"Losowanie {test_size} próbek do testu z pozostałych danych...")
        df_test = df_remaining.sample(n=test_size, random_state=42)
    else:
        print(f"Do testu pozostało {len(df_remaining)} próbek.")
        df_test = df_remaining.copy()

    print("Przetwarzanie zbioru treningowego...")
    X_train, _ = preprocess_netflow(df_train)
    y_train = torch.zeros(X_train.shape[0])

    print("Przetwarzanie zbioru testowego...")
    y_test_raw = df_test["ALERT"].apply(lambda x: 0 if x == "None" else 1)
    
    X_test, _ = preprocess_netflow(df_test)
    y_test = y_test_raw

    return (X_train, y_train), (X_test, y_test)