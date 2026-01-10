from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from CONFIG import D1_TRAINSET, D1_TRAINSET_FEATURES_LABELS, NETFLOW_V9_TRAIN, DATASET3

# todo:do configu
# 70% do nauki, 30% do testów
TEST_SIZE = 0.3
RANDOM_STATE_SEED = 66


def parse_netflow(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def preprocess_netflow(df):
    # Wykluczone kolumny
    exclude_cols = [
        "ALERT",
        "ANOMALY",
        "ID",
        "FLOW_ID",
        "IPV4_SRC_ADDR",
        "IPV4_DST_ADDR",
        "ANALYSIS_TIMESTAMP",
        "FIRST_SWITCHED",
        "LAST_SWITCHED",
        "L4_SRC_PORT",
        "PROTOCOL_MAP",
        "TOTAL_FLOWS_EXP",
        "TOTAL_PKTS_EXP",
        "TOTAL_BYTES_EXP",
    ]

    # Kolumny do logarytmizacji
    log_cols = [
        "IN_BYTES",
        "IN_PKTS",
        "OUT_BYTES",
        "OUT_PKTS",
        "FLOW_DURATION_MILLISECONDS",
        "MIN_IP_PKT_LEN",
        "MAX_IP_PKT_LEN",
        "TCP_WIN_MAX_IN",
        "TCP_WIN_MAX_OUT",
        "TCP_WIN_MIN_IN",
        "TCP_WIN_MIN_OUT",
        "TCP_WIN_MSS_IN",
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

    # Wybór danych do TESTU (Mix: Ataki + pozostała Norma)
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


# ------------------
# dataset 1
def extract_features():
    features_path = D1_TRAINSET_FEATURES_LABELS
    arr = []
    with open(features_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if ":" in line:
                feature = line.split(":")[0]
                arr.append(feature)
    
    if "anomaly" not in arr:
        arr.append("anomaly")
    return arr

def preprocess_dataset1() -> pd.DataFrame:
    features = extract_features()
    d1_path = D1_TRAINSET

    try:
        df = pd.read_csv(d1_path, header=None, names=features)
    except Exception as e:
        print(f"Błąd wczytywania CSV: {e}")
        df = pd.read_csv(d1_path, header=None)
    
    # Usuwanie duplikatów
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"Usunięto duplikaty: {initial_len} -> {len(df)}")
    
    # Konwersja etykiet: 'normal.' -> 0, reszta -> 1
    df["anomaly"] = df["anomaly"].apply(lambda x: 0 if str(x).strip() == "normal." else 1)
    
    return df

def load_dataset1() -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
    df = preprocess_dataset1()
    X = df.drop(columns=["anomaly"])
    y = df["anomaly"]

    cat_cols = ["protocol_type", "service", "flag"]
    for col in cat_cols:
        if col in X.columns:
            freq = X[col].value_counts() / len(X)
            X[col] = X[col].map(freq).astype(float)

    drop_cols = [
        "num_outbound_cmds",
        "is_host_login",
    ]

    X = X.drop(columns=[c for c in drop_cols if c in X.columns])

    log_cols = [
        "duration",
        "src_bytes", "dst_bytes", 
        "wrong_fragment", "urgent",
        "hot", "num_failed_logins", 
        "num_compromised",
        "num_root",
        "num_file_creations", 
        "num_shells", 
        "num_access_files",
        "count", "srv_count", 
        "dst_host_count", "dst_host_srv_count"
    ]
    
    for col in log_cols:
        if col in X.columns:
            X[col] = np.log1p(X[col].clip(lower=0))

    X = X.fillna(0).astype(float)

    # Podział danych
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE_SEED, stratify=y
    )

    # Filtracja zbioru treningowego -> tylko norma
    train_mask = (y_train_raw == 0)
    X_train = X_train_raw[train_mask]
    y_train = y_train_raw[train_mask]

    print(f"Dane wczytane (KDD/NSL-KDD).")
    print(f"Trening (Tylko Norma): {X_train.shape}")
    print(f"Test (Mix): {X_test.shape}")
    
    return (X_train, y_train), (X_test, y_test)



# ------------------
# dataset 3


def preprocess_dataset3() -> pd.DataFrame:
    DATASET3_PATH = DATASET3

    df = pd.read_csv(DATASET3_PATH, header=None)
    # ostatnia kolumna = etykieta
    n_cols = df.shape[1]
    feature_cols = [f"f{i}" for i in range(n_cols - 1)]
    df.columns = feature_cols + ["anomaly"]

    df["anomaly"] = df["anomaly"].astype(int)

    df = df.drop_duplicates()
    return df


def load_dataset3() -> Tuple[
    Tuple[pd.DataFrame, pd.Series],
    Tuple[pd.DataFrame, pd.Series],
]:
    df = preprocess_dataset3()

    X = df.drop(columns=["anomaly"])
    y = df["anomaly"]

    log_cols = X.columns
    for col in log_cols:
        X[col] = np.log1p(X[col].clip(lower=0))

    X = X.fillna(0).astype(float)

    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE_SEED,
        stratify=y,
    )

    mask = y_train_raw == 0
    X_train = X_train_raw[mask]
    y_train = y_train_raw[mask]
    return (X_train, y_train), (X_test, y_test)

