from datasets.datasets_init import download_data, __BASE_DIR



def test_handle():
    pass

if __name__ == '__main__':
    learning_dir = __BASE_DIR / "learning_data"
    dataset_dirs = ['dataset1_kdd99', 'dataset3', 'netflow_v9']
    for d in dataset_dirs:
        dir_path = learning_dir / d
        if not dir_path.exists():
            download_data()
            break

