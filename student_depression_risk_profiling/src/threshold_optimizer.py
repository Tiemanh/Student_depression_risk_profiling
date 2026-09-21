import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from .config import MODEL_DIR, RESULTS_DIR, FIGURES_DIR
from .utils import load_pickle, save_pickle, ensure_directories


def _false_counts(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return int(fn), int(fp)


def optimize_threshold():
    ensure_directories()
    artifacts = load_pickle(MODEL_DIR / "training_artifacts.pkl")
    y_val = artifacts["y_val"]
    y_proba = artifacts["y_proba_val"]

    rows = []
    for thr in np.arange(0.20, 0.801, 0.01):
        y_pred = (y_proba >= thr).astype(int)
        fn, fp = _false_counts(y_val, y_pred)
        rows.append({
            "threshold": round(float(thr), 2),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "false_negative": fn,
            "false_positive": fp,
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "threshold_optimization.csv", index=False)

    candidates = df[df["recall"] >= 0.85]
    if len(candidates) > 0:
        best = candidates.sort_values(["f1", "precision"], ascending=False).iloc[0]
    else:
        best = df.sort_values(["f1", "recall"], ascending=False).iloc[0]
    best_threshold = float(best["threshold"])
    save_pickle(best_threshold, MODEL_DIR / "best_threshold.pkl")

    plt.figure(figsize=(8, 4))
    plt.plot(df["threshold"], df["precision"], label="Precision")
    plt.plot(df["threshold"], df["recall"], label="Recall")
    plt.plot(df["threshold"], df["f1"], label="F1-score")
    plt.axvline(best_threshold, linestyle="--", label=f"Best={best_threshold:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold analysis on validation set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_analysis.png", dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Best threshold chọn trên validation: {best_threshold:.2f}")
    return best_threshold


if __name__ == "__main__":
    optimize_threshold()
