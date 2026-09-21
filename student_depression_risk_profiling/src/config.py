from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "student_depression_dataset.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "student_depression_clean.csv"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

TARGET_COL = "depression"
RANDOM_STATE = 42

# City bị loại vì app không dùng và dễ gây nhiễu theo khu vực.
# Work pressure và job satisfaction được giữ để hỗ trợ sinh viên vừa học vừa làm.
EXCLUDED_FEATURES = ["city"]

PROFILE_GROUPS = {
    "academic": ["academic_pressure", "cgpa", "study_satisfaction", "degree"],
    "work": ["work_pressure", "job_satisfaction", "profession"],
    "financial": ["financial_stress"],
    "lifestyle": ["sleep_duration", "dietary_habits", "work_study_hours"],
    "psychological": ["family_history", "suicidal_thoughts"],
}

PROFILE_LABELS = {
    "academic": "Học tập",
    "work": "Đi làm/công việc",
    "financial": "Tài chính",
    "lifestyle": "Lối sống/giấc ngủ",
    "psychological": "Tâm lý/an toàn",
}

RISK_LEVELS = {
    "low": "Nguy cơ thấp",
    "medium": "Cần theo dõi thêm",
    "high": "Nguy cơ cao",
}
