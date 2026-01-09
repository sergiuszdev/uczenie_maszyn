from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from CONFIG import D1_TRAINSET, D1_TRAINSET_FEATURES_LABELS, NETFLOW_V9_TRAIN

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
        skip = f.readline()
        for line in f.readlines():
            feature = line.split(":")[0]
            arr.append(feature)
    # brakuje kolumny o ataku
    arr.append("anomaly")
    return arr


def preprocess_dataset1() -> pd.DataFrame:
    features = extract_features()
    d1_path = D1_TRAINSET

    """
    ['duration', 'protocol_type', 'service',
    'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'anomaly']
    """

    df = pd.read_csv(d1_path, header=None, names=features)
    df["anomaly"] = df["anomaly"].apply(lambda x: 0 if x == "normal." else 1)
    return df


def load_dataset1() -> Tuple[
    Tuple[pd.DataFrame, pd.Series],
    Tuple[pd.DataFrame, pd.Series],
]:
    df = preprocess_dataset1()
    X = df.drop(columns=["anomaly"])
    y = df["anomaly"]

    for col in ["protocol_type", "service", "flag"]:
        freq = X[col].value_counts() / len(X)
        X[col] = X[col].map(freq)

    drop_cols = [
        "num_outbound_cmds",
        "is_guest_login",
        "is_host_login",
        "root_shell",
        "su_attempted",
    ]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])

    numeric_cols = ["src_bytes", "dst_bytes", "count", "srv_count", "num_failed_logins"]
    for col in numeric_cols:
        if col in X.columns:
            X[col] = np.log1p(X[col])

    X = X.fillna(0).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE_SEED, stratify=y
    )

    return (X_train, y_train), (X_test, y_test)
