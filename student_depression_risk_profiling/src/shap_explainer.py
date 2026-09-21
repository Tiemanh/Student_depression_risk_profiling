import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import MODEL_DIR, RESULTS_DIR, FIGURES_DIR, PROFILE_GROUPS, PROFILE_LABELS
from .utils import load_pickle, ensure_directories
from .feature_labels import feature_to_vietnamese


def _clean_feature_name(name: str) -> str:
    return str(name).replace("num__", "").replace("cat__", "")


def _get_shap_matrix(shap_values):
    if isinstance(shap_values, list):
        return np.asarray(shap_values[-1])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[:, :, -1]
    return arr


def build_shap_global_explanation(max_samples=1000):
    """Tạo giải thích toàn cục bằng SHAP cho LightGBM."""
    ensure_directories()
    try:
        import shap
        preprocessor = load_pickle(MODEL_DIR / "preprocessor.pkl")
        model = load_pickle(MODEL_DIR / "lightgbm_model.pkl")
        artifacts = load_pickle(MODEL_DIR / "training_artifacts.pkl")
        X_test = artifacts["X_test"].copy()
        X_sample = X_test.sample(max_samples, random_state=42) if len(X_test) > max_samples else X_test

        X_t = preprocessor.transform(X_sample)
        feature_raw = [_clean_feature_name(f) for f in preprocessor.get_feature_names_out()]
        feature_vi = [feature_to_vietnamese(f) for f in feature_raw]

        explainer = shap.TreeExplainer(model)
        shap_values = _get_shap_matrix(explainer.shap_values(X_t))
        mean_abs = np.abs(shap_values).mean(axis=0)

        imp = pd.DataFrame({
            "feature_raw": feature_raw,
            "feature_vi": feature_vi,
            "mean_abs_shap": mean_abs,
            "mean_abs_shap_label": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False)
        imp.to_csv(RESULTS_DIR / "shap_global_importance.csv", index=False)

        top = imp.head(12).iloc[::-1]
        plt.figure(figsize=(6.2, 4.2))
        plt.barh(top["feature_vi"], top["mean_abs_shap"])
        plt.title("SHAP: các yếu tố quan trọng nhất của mô hình")
        plt.xlabel("Độ quan trọng trung bình |SHAP|")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_global_bar.png", dpi=150, bbox_inches="tight")
        plt.close()

        try:
            shap.summary_plot(shap_values, X_t, feature_names=feature_vi, show=False, max_display=12)
            plt.title("SHAP summary: ảnh hưởng của feature đến nguy cơ")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Cảnh báo: không vẽ được beeswarm SHAP: {e}")

        profile_rows = []
        for key, keywords in PROFILE_GROUPS.items():
            mask = [any(k in f.lower() for k in keywords) for f in feature_raw]
            score = float(mean_abs[mask].sum()) if any(mask) else 0.0
            profile_rows.append({
                "profile": key,
                "label": PROFILE_LABELS.get(key, key),
                "mean_abs_shap_sum": score,
            })
        profile_df = pd.DataFrame(profile_rows).sort_values("mean_abs_shap_sum", ascending=False)
        profile_df.to_csv(RESULTS_DIR / "shap_profile_importance.csv", index=False)

        plt.figure(figsize=(5.8, 3.6))
        plt.bar(profile_df["label"], profile_df["mean_abs_shap_sum"])
        plt.xticks(rotation=15, ha="right")
        plt.title("SHAP theo nhóm hồ sơ nguy cơ")
        plt.ylabel("Tổng độ quan trọng |SHAP|")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_profile_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Đã tạo SHAP global explanation.")
    except Exception as e:
        print(f"Cảnh báo: bỏ qua SHAP do lỗi: {e}")


if __name__ == "__main__":
    build_shap_global_explanation()
