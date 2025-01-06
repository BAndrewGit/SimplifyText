import json
from typing import List, Dict

def load_json_dataset(path: str) -> List[Dict]:

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_train_dev_datasets(train_path: str, dev_path: str):
    train_data = load_json_dataset(train_path)
    dev_data = load_json_dataset(dev_path)
    return train_data, dev_data

if __name__ == "__main__":
    # Test
    train_data, dev_data = load_train_dev_datasets("train.json", "dev.json")
    print(f"Train size: {len(train_data)}")
    print(f"Dev size: {len(dev_data)}")
    print("Exemplu train:", train_data[0])
