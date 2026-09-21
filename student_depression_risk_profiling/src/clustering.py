import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from .config import PROCESSED_DATA_PATH, MODEL_DIR, RESULTS_DIR, FIGURES_DIR, TARGET_COL, RANDOM_STATE
from .utils import load_pickle, save_pickle, ensure_directories

CLUSTER_BASE_FEATURES = [
    "risk_probability", "academic_pressure", "work_pressure", "job_satisfaction",
    "financial_stress", "study_satisfaction", "work_study_hours", "cgpa",
    "sleep_duration", "dietary_habits", "family_history", "suicidal_thoughts",
]


def _name_cluster(row, all_summary):
    risk = row.get("avg_risk_probability", 0)
    if risk <= all_summary["avg_risk_probability"].quantile(0.25):
        return "Nhóm nguy cơ thấp"
    ap = row.get("avg_academic_pressure", np.nan)
    fs = row.get("avg_financial_stress", np.nan)
    wp = row.get("avg_work_pressure", np.nan)
    wh = row.get("avg_work_study_hours", np.nan)
    ss = row.get("avg_study_satisfaction", np.nan)
    if pd.notna(wp) and wp >= all_summary.get("avg_work_pressure", pd.Series([wp])).median() and risk > all_summary["avg_risk_probability"].median():
        return "Nhóm áp lực đi làm"
    if pd.notna(ap) and pd.notna(ss) and ap >= all_summary.get("avg_academic_pressure", pd.Series([ap])).median() and ss <= all_summary.get("avg_study_satisfaction", pd.Series([ss])).median():
        return "Nhóm áp lực học tập"
    if pd.notna(fs) and fs >= all_summary.get("avg_financial_stress", pd.Series([fs])).median():
        return "Nhóm áp lực tài chính"
    if pd.notna(wh) and wh >= all_summary.get("avg_work_study_hours", pd.Series([wh])).median():
        return "Nhóm lối sống/giấc ngủ"
    if risk >= all_summary["avg_risk_probability"].quantile(0.75):
        return "Nhóm nguy cơ tổng hợp"
    return "Nhóm hồ sơ hỗn hợp"


def run_clustering():
    ensure_directories()
    df = pd.read_csv(PROCESSED_DATA_PATH)
    pipeline = load_pickle(MODEL_DIR / "full_pipeline.pkl")
    metadata = load_pickle(MODEL_DIR / "feature_metadata.pkl")
    X_raw = df[metadata["raw_feature_columns"]].copy()
    df["risk_probability"] = pipeline.predict_proba(X_raw)[:, 1]

    selected = [c for c in CLUSTER_BASE_FEATURES if c in df.columns]
    cluster_df = df[selected].copy()
    num_cols = cluster_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in cluster_df.columns if c not in num_cols]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat_cols),
    ])
    X_t = preprocessor.fit_transform(cluster_df)

    best_k, best_score, best_model = 2, -1, None
    for k in range(2, 7):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_t)
        score = silhouette_score(X_t, labels) if len(set(labels)) > 1 else -1
        if score > best_score:
            best_k, best_score, best_model = k, score, model
    labels = best_model.predict(X_t)
    df["cluster"] = labels

    rows = []
    for c in sorted(df["cluster"].unique()):
        part = df[df["cluster"] == c]
        row = {"cluster_id": int(c), "count": len(part), "avg_risk_probability": float(part["risk_probability"].mean()), "silhouette_score": float(best_score), "best_k": int(best_k)}
        for f in ["academic_pressure", "work_pressure", "financial_stress", "study_satisfaction", "work_study_hours", "cgpa"]:
            if f in part.columns:
                row[f"avg_{f}"] = pd.to_numeric(part[f], errors="coerce").mean()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["cluster_name"] = summary.apply(lambda r: _name_cluster(r, summary), axis=1)
    
    def _describe(row):
        name = row.get("cluster_name", "Nhóm hồ sơ hỗn hợp")
        risk_pct = row.get("avg_risk_probability", 0) * 100
        desc = f"Đây là nhóm sinh viên có xác suất nguy cơ trung bình khoảng {risk_pct:.1f}%. "
        if name == "Nhóm nguy cơ thấp":
            return desc + "Các yếu tố áp lực nhìn chung thấp hơn các nhóm còn lại."
        if name == "Nhóm áp lực học tập":
            return desc + "Đặc trưng nổi bật là áp lực học tập cao hoặc mức hài lòng với việc học thấp."
        if name == "Nhóm áp lực đi làm":
            return desc + "Đặc trưng nổi bật là áp lực công việc/việc làm thêm hoặc sự cân bằng học–làm chưa tốt."
        if name == "Nhóm áp lực tài chính":
            return desc + "Đặc trưng nổi bật là áp lực tài chính cao."
        if name == "Nhóm lối sống/giấc ngủ":
            return desc + "Đặc trưng nổi bật là thời gian học/làm dài, giấc ngủ hoặc thói quen sinh hoạt chưa ổn định."
        if name == "Nhóm nguy cơ tổng hợp":
            return desc + "Nhiều nhóm yếu tố cùng ở mức đáng chú ý, cần xem xét kết hợp học tập, tài chính, lối sống và tâm lý."
        return desc + "Nhóm này có đặc điểm pha trộn, không nghiêng hẳn về một nguyên nhân duy nhất."

    summary["cluster_description"] = summary.apply(_describe, axis=1)
    summary.to_csv(RESULTS_DIR / "cluster_summary.csv", index=False)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_t)
    plt.figure(figsize=(6, 4))
    plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=8)
    plt.title("Trực quan hóa nhóm sinh viên tương đồng bằng PCA")
    plt.xlabel("Thành phần chính 1")
    plt.ylabel("Thành phần chính 2")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_clusters.png", dpi=160, bbox_inches="tight")
    plt.close()

    radar_cols = [c for c in ["avg_risk_probability", "avg_academic_pressure", "avg_work_pressure", "avg_financial_stress", "avg_study_satisfaction"] if c in summary.columns]
    if radar_cols:
        plot_df = summary[["cluster_name"] + radar_cols].set_index("cluster_name").fillna(0)
        norm = (plot_df - plot_df.min()) / (plot_df.max() - plot_df.min()).replace(0, 1)
        norm.plot(kind="bar", figsize=(7, 4))
        plt.title("Tóm tắt đặc trưng các nhóm tương đồng")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "cluster_radar_chart.png", dpi=160, bbox_inches="tight")
        plt.close()

    cluster_metadata = {"selected_features": selected, "num_cols": num_cols, "cat_cols": cat_cols, "summary": summary}
    save_pickle(preprocessor, MODEL_DIR / "cluster_scaler.pkl")
    save_pickle(best_model, MODEL_DIR / "cluster_model.pkl")
    save_pickle(cluster_metadata, MODEL_DIR / "cluster_metadata.pkl")
    print("Đã chạy clustering.")


def predict_cluster(raw_input_df, risk_probability):
    try:
        preprocessor = load_pickle(MODEL_DIR / "cluster_scaler.pkl")
        model = load_pickle(MODEL_DIR / "cluster_model.pkl")
        meta = load_pickle(MODEL_DIR / "cluster_metadata.pkl")
        row = raw_input_df.copy()
        row["risk_probability"] = risk_probability
        for c in meta["selected_features"]:
            if c not in row.columns:
                row[c] = np.nan
        X = row[meta["selected_features"]]
        cluster_id = int(model.predict(preprocessor.transform(X))[0])
        summary = meta["summary"]
        info = summary[summary["cluster_id"] == cluster_id]
        if len(info):
            return {"cluster_id": cluster_id, "cluster_name": str(info.iloc[0]["cluster_name"]), "cluster_description": str(info.iloc[0]["cluster_description"])}
    except Exception as e:
        return {"cluster_id": None, "cluster_name": "Chưa xác định", "cluster_description": f"Không xác định được nhóm tương đồng: {e}"}
    return {"cluster_id": cluster_id, "cluster_name": "Nhóm hồ sơ hỗn hợp", "cluster_description": "Không có mô tả chi tiết."}


if __name__ == "__main__":
    run_clustering()
