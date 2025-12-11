import pandas as pd
import torch
import numpy as np

from CONFIG import NETFLOW_V9_TRAIN, NETFLOW_V9_TEST

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

def get_training_data():
    print("Ładowanie i przygotowywanie zbioru treningowego...")
    df = parse_netflow(NETFLOW_V9_TRAIN)

    # Przygotuj kolumny
    if "ALERT" in df.columns:
        df["ALERT"] = df["ALERT"].fillna("None")
    else:
        df["ALERT"] = "None"
        
    if "ANOMALY" in df.columns:
        df["ANOMALY"] = df["ANOMALY"].fillna(0) 
    else:
        df["ANOMALY"] = 0

    # Definicja ruchu normalnego:
    # Musi mieć ALERT == "None" oraz ANOMALY == 0
    clean_mask = (df["ALERT"] == "None") & (df["ANOMALY"] == 0)
    
    df_normal_traffic = df[clean_mask]
    
    print(f"Oryginalny rozmiar: {len(df)}. Pozostało: {len(df_normal_traffic)}")

    # Sampling
    sample_size = 500000
    if len(df_normal_traffic) > sample_size:
        print(f"Losowanie {sample_size} próbek...")
        df_normal_traffic = df_normal_traffic.sample(n=sample_size, random_state=42)

    X, feature_cols = preprocess_netflow(df_normal_traffic)
    y = torch.zeros(X.shape[0])
    return X, y

def get_test_data():
    print("Ładowanie zbioru testowego...")
    df = parse_netflow(NETFLOW_V9_TEST)

    # Usuwamy wiersze, gdzie ANOMALY jest puste
    initial_len = len(df)
    df = df.dropna(subset=["ANOMALY"])
    dropped_len = initial_len - len(df)
    if dropped_len > 0:
        print(f"Usunięto {dropped_len} wierszy z testu z powodu braku etykiety ANOMALY")

    # Konwersja ANOMALY na 0 i 1
    # 0 lub "None" to OK, wszystko inne to atak
    df["ANOMALY"] = df["ANOMALY"].apply(lambda x: 0 if str(x) in ["None", "0", "0.0"] else 1)
    y = df["ANOMALY"]

    X, _ = preprocess_netflow(df)
    return X, y