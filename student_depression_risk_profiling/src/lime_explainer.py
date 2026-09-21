import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import MODEL_DIR, RESULTS_DIR, FIGURES_DIR
from .utils import load_pickle, ensure_directories
from .feature_labels import feature_to_vietnamese, explanation_direction


def _clean_feature_name(name: str) -> str:
    s = str(name).replace("num__", "").replace("cat__", "")
    for base in [
        "sleep_duration", "dietary_habits", "family_history", "suicidal_thoughts",
        "gender", "degree", "profession"
    ]:
        if s.startswith(base + "_"):
            return f"{base} = {s[len(base)+1:]}"
    return s


def explain_instance(raw_input_df: pd.DataFrame, num_features: int = 10):
    """Giải thích một mẫu bằng LIME, trả về cả tên kỹ thuật và tên tiếng Việt."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
        preprocessor = load_pickle(MODEL_DIR / "preprocessor.pkl")
        model = load_pickle(MODEL_DIR / "lightgbm_model.pkl")
        artifacts = load_pickle(MODEL_DIR / "training_artifacts.pkl")
        X_train = artifacts["X_train"]
        X_train_t = preprocessor.transform(X_train)
        x_t = preprocessor.transform(raw_input_df)
        feature_names_raw = preprocessor.get_feature_names_out()
        feature_names_clean = [_clean_feature_name(f) for f in feature_names_raw]
        feature_names_vi = [feature_to_vietnamese(f) for f in feature_names_clean]

        explainer = LimeTabularExplainer(
            np.asarray(X_train_t),
            feature_names=feature_names_vi,
            class_names=["Chưa thấy dấu hiệu rõ", "Có dấu hiệu nguy cơ"],
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )
        exp = explainer.explain_instance(
            np.asarray(x_t)[0], model.predict_proba, labels=[1], num_features=num_features
        )
        rows = []
        for f_vi, w in exp.as_list(label=1):
            rows.append({
                "feature": f_vi,
                "feature_vi": f_vi,
                "weight": float(w),
                "direction": explanation_direction(float(w)),
            })
        return rows
    except Exception as e:
        raise RuntimeError(f"Không tạo được LIME explanation: {e}")


def build_lime_example():
    ensure_directories()
    try:
        pipeline = load_pickle(MODEL_DIR / "full_pipeline.pkl")
        artifacts = load_pickle(MODEL_DIR / "training_artifacts.pkl")
        X_test = artifacts["X_test"]
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        idx = int(np.argmax(y_proba))
        sample = X_test.iloc[[idx]]
        explanation = explain_instance(sample, num_features=10)
        df = pd.DataFrame(explanation)
        df.to_csv(RESULTS_DIR / "risk_profile_examples.csv", index=False)

        plot_df = df.iloc[::-1]
        colors = ["#d62728" if w > 0 else "#2ca02c" for w in plot_df["weight"]]
        plt.figure(figsize=(6.2, 3.8))
        plt.barh(plot_df["feature_vi"], plot_df["weight"], color=colors)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.title("LIME: yếu tố ảnh hưởng đến dự đoán cá nhân")
        plt.xlabel("Trọng số LIME")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "lime_example.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Đã tạo ví dụ LIME.")
    except Exception as e:
        print(f"Cảnh báo: bỏ qua LIME example vì lỗi: {e}")


if __name__ == "__main__":
    build_lime_example()
