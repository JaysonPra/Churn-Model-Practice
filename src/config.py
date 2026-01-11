from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RAW_DATA_PATH = DATA_DIR / "raw" / DATA_FILE
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "cleaned_data.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
EXP_CONFIG_DIR = PROJECT_ROOT / "config"