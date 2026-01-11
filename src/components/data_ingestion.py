import kagglehub
from kagglehub import KaggleDatasetAdapter
from config import RAW_DATA_PATH, DATA_FILE
import pandas as pd
import shutil

try:
    print("Downloading dataset from Kaggle...")

    cache_data = kagglehub.dataset_download("blastchar/telco-customer-churn")
    downloaded_file = f"{cache_data}/{DATA_FILE}"

    df = pd.read_csv(downloaded_file)

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)

    print(f"Data File successfully installed to {RAW_DATA_PATH}")

    print(f"Clearing up cache at {cache_data}")
    shutil.rmtree(cache_data)

    print("Cache Cleared Successfully...")

except Exception as e:
    print(f"Ingestion failed: {e}")
