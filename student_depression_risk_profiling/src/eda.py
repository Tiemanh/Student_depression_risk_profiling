import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .config import PROCESSED_DATA_PATH, FIGURES_DIR, TARGET_COL
from .utils import ensure_directories


def _safe_savefig(name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=160, bbox_inches="tight")
    plt.close()


def _risk_rate_plot(df, col, filename, title):
    if col not in df.columns:
        print(f"Bỏ qua {filename}: thiếu cột {col}")
        return
    tmp = df[[col, TARGET_COL]].copy()
    tmp[col] = tmp[col].astype("object").where(tmp[col].notna(), "Missing")
    # Ép nhãn x về string để tránh lỗi matplotlib khi lẫn float và NaN.
    tmp[col] = tmp[col].astype(str)
    tmp = tmp.groupby(col, dropna=False)[TARGET_COL].mean().reset_index()
    tmp = tmp.sort_values(col)
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(tmp)), tmp[TARGET_COL])
    plt.xticks(range(len(tmp)), tmp[col], rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel("Tỷ lệ Depression = 1")
    _safe_savefig(filename)


def run_eda():
    ensure_directories()
    df = pd.read_csv(PROCESSED_DATA_PATH)

    plt.figure(figsize=(5, 4))
    counts = df[TARGET_COL].value_counts().sort_index()
    plt.bar([str(x) for x in counts.index], counts.values)
    plt.title("Phân bố nhãn Depression")
    plt.xlabel("Depression")
    plt.ylabel("Số lượng")
    _safe_savefig("label_distribution.png")

    _risk_rate_plot(df, "academic_pressure", "academic_pressure_risk.png", "Tỷ lệ nguy cơ theo Academic Pressure")
    _risk_rate_plot(df, "financial_stress", "financial_stress_risk.png", "Tỷ lệ nguy cơ theo Financial Stress")
    _risk_rate_plot(df, "sleep_duration", "sleep_duration_risk.png", "Tỷ lệ nguy cơ theo Sleep Duration")
    _risk_rate_plot(df, "study_satisfaction", "study_satisfaction_risk.png", "Tỷ lệ nguy cơ theo Study Satisfaction")
    _risk_rate_plot(df, "work_pressure", "work_pressure_risk.png", "Tỷ lệ nguy cơ theo Work Pressure")

    if {"academic_pressure", "financial_stress"}.issubset(df.columns):
        tmp = df[["academic_pressure", "financial_stress", TARGET_COL]].copy()
        tmp["academic_pressure"] = tmp["academic_pressure"].astype(str)
        tmp["financial_stress"] = tmp["financial_stress"].astype(str)
        pivot = tmp.pivot_table(index="academic_pressure", columns="financial_stress", values=TARGET_COL, aggfunc="mean")
        plt.figure(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues")
        plt.title("Interaction: Academic Pressure × Financial Stress")
        _safe_savefig("interaction_heatmap.png")

    print(f"Đã lưu EDA figures tại: {FIGURES_DIR}")


if __name__ == "__main__":
    run_eda()
