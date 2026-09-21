import re
import pandas as pd
from .config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET_COL
from .utils import ensure_directories

_ALIAS_MAP = {
    "have_you_ever_had_suicidal_thoughts": "suicidal_thoughts",
    "have_you_ever_had_suicidal_thoughts_": "suicidal_thoughts",
    "suicidal_thoughts": "suicidal_thoughts",
    "family_history_of_mental_illness": "family_history",
    "family_history": "family_history",
    "work_study_hours": "work_study_hours",
    "work_or_study_hours": "work_study_hours",
    "study_hours": "work_study_hours",
    "academic_pressure": "academic_pressure",
    "work_pressure": "work_pressure",
    "job_satisfaction": "job_satisfaction",
    "study_satisfaction": "study_satisfaction",
    "financial_stress": "financial_stress",
    "sleep_duration": "sleep_duration",
    "dietary_habits": "dietary_habits",
    "depression": "depression",
}


def _to_snake(name: str) -> str:
    name = str(name).strip().lower()
    name = name.replace("/", "_")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return _ALIAS_MAP.get(name, name)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_to_snake(c) for c in df.columns]
    # Nếu bị trùng cột sau normalize, giữ cột đầu tiên.
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def _map_yes_no(series: pd.Series) -> pd.Series:
    def conv(v):
        if pd.isna(v):
            return pd.NA
        s = str(v).strip().lower()
        if s in {"yes", "y", "true", "1", "có", "co"}:
            return 1
        if s in {"no", "n", "false", "0", "không", "khong"}:
            return 0
        try:
            return int(float(s))
        except Exception:
            return pd.NA
    return series.apply(conv)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_column_names(df)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Không tìm thấy cột target '{TARGET_COL}'. Các cột hiện có: {list(df.columns)}")

    # Chuẩn hóa text object để giảm lỗi do khoảng trắng thừa.
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].str.lower().isin(["nan", "none", "null", ""]), col] = pd.NA

    df[TARGET_COL] = _map_yes_no(df[TARGET_COL])
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def load_and_clean_data() -> pd.DataFrame:
    ensure_directories()
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dataset tại {RAW_DATA_PATH}. "
            "Hãy đặt file CSV vào data/raw/student_depression_dataset.csv"
        )
    df = pd.read_csv(RAW_DATA_PATH)
    df = clean_dataset(df)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Shape sau làm sạch: {df.shape}")
    print("Danh sách cột:", list(df.columns))
    print("Missing values:")
    print(df.isna().sum().sort_values(ascending=False))
    print(f"Đã lưu dữ liệu sạch tại: {PROCESSED_DATA_PATH}")
    return df


if __name__ == "__main__":
    load_and_clean_data()
