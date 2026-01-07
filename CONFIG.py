from pathlib import Path

__BASE_DIR = Path(__file__).parent

NETFLOW_V9_TRAIN = __BASE_DIR / "learning_data" / "netflow_v9" / "train_net.csv"
NETFLOW_V9_TEST = __BASE_DIR / "learning_data" / "netflow_v9" / "test_net.csv"


# dataset1
KDD99_DIR = __BASE_DIR / "learning_data" / "dataset1_kdd99"
# The correct data is now in kddcup.data.gz and kddcup-data_10_percent.gz.
D1_TRAINSET = KDD99_DIR / "kddcup.data"
D1_TRAINSET_FEATURES_LABELS = KDD99_DIR / "kddcup.names"
