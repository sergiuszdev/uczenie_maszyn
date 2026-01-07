# pobranie danych z kaggla
import gzip
import shutil
from pathlib import Path

import kagglehub


def move_subdirs(src, dst):
    src = Path(src)
    dst = Path(dst)
    for item in src.iterdir():
        shutil.move(str(item), dst / item.name)


def extract_gzs(directory: Path):
    for gz_file in directory.iterdir():
        if gz_file.is_file() and gz_file.suffix == ".gz":
            __extract_gz(gz_file)


def __extract_gz(file_path: Path):
    out = file_path.with_suffix("")
    with gzip.open(file_path, "rb") as f_in:
        with open(out, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    # remove/unlink
    file_path.unlink()


__BASE_DIR = Path(__file__).parent
DATASETS_DIR = __BASE_DIR / "learning_data"


# dataset1
cached_dataset1 = kagglehub.dataset_download("galaxyh/kdd-cup-1999-data")
DATASET_KDD = DATASETS_DIR / "dataset1_kdd99"
DATASET_KDD.mkdir(parents=True, exist_ok=True)
move_subdirs(cached_dataset1, DATASET_KDD)
extract_gzs(DATASET_KDD)
print(f"KDD 99 moved to {DATASET_KDD}")

cached_dataset2 = kagglehub.dataset_download(
    "ashtcoder/network-data-schema-in-the-netflow-v9-format"
)
DATASET_NETFLOWV9 = DATASETS_DIR / "netflow_v9"
DATASET_NETFLOWV9.mkdir(parents=True, exist_ok=True)
move_subdirs(cached_dataset2, DATASET_NETFLOWV9)
print(f"netflowv9 moved to {DATASET_NETFLOWV9}")

# trzeciego nie mogę znaleźć na kagglu
