from pathlib import Path
import joblib
import pandas as pd
from .config import DATA_DIR, MODEL_DIR, RESULTS_DIR, FIGURES_DIR


def ensure_directories():
    for p in [DATA_DIR / "raw", DATA_DIR / "processed", MODEL_DIR, RESULTS_DIR, FIGURES_DIR]:
        Path(p).mkdir(parents=True, exist_ok=True)


def save_pickle(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_pickle(path):
    return joblib.load(path)


def print_step(message: str):
    print("\n" + "=" * 80)
    print(message)
    print("=" * 80)


def normalize_text_value(x):
    if pd.isna(x):
        return x
    return str(x).strip()


def file_exists(path):
    return Path(path).exists()
