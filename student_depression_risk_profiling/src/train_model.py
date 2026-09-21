import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from lightgbm import LGBMClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

from .config import (
    PROCESSED_DATA_PATH, MODEL_DIR, RESULTS_DIR, TARGET_COL,
    RANDOM_STATE, EXCLUDED_FEATURES
)
from .utils import ensure_directories, save_pickle


def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_negative": int(fn),
        "false_positive": int(fp),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
    except Exception:
        out["roc_auc"] = np.nan
    return out


def _build_preprocessor(numeric_features, categorical_features):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_ohe()),
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, categorical_features),
    ], remainder="drop")


def train_models():
    ensure_directories()
    df = pd.read_csv(PROCESSED_DATA_PATH)
    drop_cols = [TARGET_COL] + [c for c in EXCLUDED_FEATURES if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COL].astype(int)

    # Chia 70/15/15 để tránh leakage khi tối ưu threshold.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X_train.columns if c not in numeric_features]
    preprocessor = _build_preprocessor(numeric_features, categorical_features)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            random_state=RANDOM_STATE, class_weight="balanced", n_jobs=1, verbose=-1
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=RANDOM_STATE, tree_method="hist", n_jobs=1, verbosity=0
        )

    rows = []
    fitted_pipelines = {}
    for name, model in models.items():
        print(f"Đang train: {name}")
        try:
            pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            y_proba = pipe.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            m = _metrics(y_test, y_pred, y_proba)
            m["model"] = name
            rows.append(m)
            fitted_pipelines[name] = pipe
            if name == "XGBoost":
                save_pickle(pipe, MODEL_DIR / "xgboost_pipeline.pkl")
        except Exception as e:
            rows.append({"model": name, "error": str(e)})
            print(f"Bỏ qua {name} do lỗi: {e}")

    comparison = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    if "LightGBM" not in fitted_pipelines:
        raise RuntimeError("LightGBM train thất bại, không thể tạo final model.")

    final_pipeline = fitted_pipelines["LightGBM"]
    final_preprocessor = final_pipeline.named_steps["preprocessor"]
    final_model = final_pipeline.named_steps["model"]

    try:
        transformed_feature_names = final_preprocessor.get_feature_names_out().tolist()
    except Exception:
        transformed_feature_names = []

    y_proba_val = final_pipeline.predict_proba(X_val)[:, 1]
    y_proba_test = final_pipeline.predict_proba(X_test)[:, 1]

    feature_metadata = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "raw_feature_columns": X.columns.tolist(),
        "target_col": TARGET_COL,
        "excluded_features": EXCLUDED_FEATURES,
        "categorical_values": {c: sorted([str(v) for v in X_train[c].dropna().unique().tolist()]) for c in categorical_features},
        "transformed_feature_names": transformed_feature_names,
    }
    training_artifacts = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_proba_val": y_proba_val,
        "y_proba_test": y_proba_test,
        "transformed_feature_names": transformed_feature_names,
    }
    split_summary = pd.DataFrame([
        {"split": "train", "n_rows": len(X_train), "positive_rate": y_train.mean()},
        {"split": "validation", "n_rows": len(X_val), "positive_rate": y_val.mean()},
        {"split": "test", "n_rows": len(X_test), "positive_rate": y_test.mean()},
    ])
    split_summary.to_csv(RESULTS_DIR / "data_split_summary.csv", index=False)

    save_pickle(final_pipeline, MODEL_DIR / "full_pipeline.pkl")
    save_pickle(final_preprocessor, MODEL_DIR / "preprocessor.pkl")
    save_pickle(final_model, MODEL_DIR / "lightgbm_model.pkl")
    save_pickle(feature_metadata, MODEL_DIR / "feature_metadata.pkl")
    save_pickle(training_artifacts, MODEL_DIR / "training_artifacts.pkl")

    print("Đã train xong. Chia dữ liệu:")
    print(split_summary)
    print("Đã lưu model LightGBM và artifacts.")
    return comparison


if __name__ == "__main__":
    train_models()
