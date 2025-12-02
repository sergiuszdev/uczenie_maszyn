import pandas as pd
import torch

from CONFIG import NETFLOW_V9_TRAIN, NETFLOW_V9_TEST


def parse_netflow(csv_path):
    df = pd.read_csv(csv_path)
    return df


# przerobienie danych na użyteczne
def preprocess_netflow(df):

    # niepotrzebne przy uczeniu modelu
    exclude_cols = [
        "ALERT", "ID", "FLOW_ID", "ANALYSIS_TIMESTAMP",
        "IPV4_SRC_ADDR", "IPV4_DST_ADDR"
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].copy()

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.factorize(X[col])[0]

    X = X.fillna(0)
    return X, feature_cols


def get_training_data():
    df = parse_netflow(NETFLOW_V9_TRAIN)

    # w test_net nie ma ALERT
    if "ALERT" in df.columns:
        df["ALERT"] = df["ALERT"].fillna("None")
        # niepotrzebne są dane jaki atak, a czy nastąpił
        # lepiej wziąć alert bo anomaly ma puste pola (??)
        df["ALERT"] = df["ALERT"].apply(lambda x: 0 if x == "None" else 1)

    df_normal_traffic = df[df["ALERT"] == 0]

    X, feature_cols = preprocess_netflow(df_normal_traffic)
    y = torch.zeros(X.shape[0])
    return X, y


def get_test_data():
    df = parse_netflow(NETFLOW_V9_TEST)
    X, feature_cols = preprocess_netflow(df)

    if "ALERT" in df.columns:
        df["ALERT"] = df["ALERT"].fillna("None")
        df["ALERT"] = df["ALERT"].apply(lambda x: 0 if x == "None" else 1)
        y = df["ALERT"]
    else:
        y = None
    return X, y

