import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)
from .config import MODEL_DIR, RESULTS_DIR, FIGURES_DIR
from .utils import load_pickle, ensure_directories


def evaluate_final_model():
    ensure_directories()
    pipeline = load_pickle(MODEL_DIR / "full_pipeline.pkl")
    threshold = load_pickle(MODEL_DIR / "best_threshold.pkl")
    artifacts = load_pickle(MODEL_DIR / "training_artifacts.pkl")
    X_test, y_test = artifacts["X_test"], artifacts["y_test"]
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {
        "evaluation_split": "test",
        "threshold_selected_on": "validation",
        "best_threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "false_negative": int(fn),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "true_positive": int(tp),
    }
    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "final_metrics.csv", index=False)

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix - Test set")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve - Test set")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=160, bbox_inches="tight")
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, y_proba)
    plt.title("Precision-Recall Curve - Test set")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pr_curve.png", dpi=160, bbox_inches="tight")
    plt.close()

    comp_path = RESULTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        if {"model", "f1", "recall"}.issubset(comp.columns):
            plot_df = comp.dropna(subset=["f1"])
            plt.figure(figsize=(9, 4))
            plt.bar(plot_df["model"].astype(str), plot_df["f1"])
            plt.xticks(rotation=25, ha="right")
            plt.ylim(0, 1)
            plt.title("Model comparison by F1-score")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=160, bbox_inches="tight")
            plt.close()

    print("Đánh giá cuối trên test set:")
    print(pd.DataFrame([metrics]).T)
    return metrics


if __name__ == "__main__":
    evaluate_final_model()
