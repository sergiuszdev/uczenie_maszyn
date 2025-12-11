from pathlib import Path

__BASE_DIR = Path(__file__).parent

NETFLOW_V9_TRAIN = __BASE_DIR / "learning_data" / "netflow_v9" / "train_net.csv"
NETFLOW_V9_TEST = __BASE_DIR / "learning_data" / "netflow_v9" / "test_net.csv"