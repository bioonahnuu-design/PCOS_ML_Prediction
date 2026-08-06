"""Reproducibly train and evaluate binary PCOS classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pcos_utils import read_dataset, split_features_target

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "PCOS_data_without_infertility.xlsx"
RANDOM_STATE = 42


def evaluate(model: Pipeline, X_test, y_test) -> dict:
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, prediction), 4),
        "precision_weighted": round(precision_score(
            y_test, prediction, average="weighted", zero_division=0
        ), 4),
        "recall_weighted": round(recall_score(
            y_test, prediction, average="weighted", zero_division=0
        ), 4),
        "f1_weighted": round(f1_score(
            y_test, prediction, average="weighted", zero_division=0
        ), 4),
        "f1_macro": round(f1_score(
            y_test, prediction, average="macro", zero_division=0
        ), 4),
        "roc_auc_ovr_weighted": round(roc_auc_score(
            y_test, probability, multi_class="ovr", average="weighted"
        ), 4),
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=[0, 1, 2]).tolist(),
    }


def main() -> None:
    features, target = split_features_target(read_dataset(DATASET_PATH))
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=RANDOM_STATE, stratify=target
    )
    models = {
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_STATE,
                class_weight="balanced", n_jobs=-1
            )),
        ]),
        "svm": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf", probability=True,
                class_weight="balanced", random_state=RANDOM_STATE
            )),
        ]),
    }
    metrics = {
        "dataset_rows": int(len(features)),
        "feature_count": int(features.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": 0.20,
        "random_state": RANDOM_STATE,
        "derived_target_distribution": {
            str(key): int(value) for key, value in target.value_counts().sort_index().items()
        },
        "models": {},
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics["models"][name] = evaluate(model, X_test, y_test)
        joblib.dump(model, BASE_DIR / f"{name}_pcos.pkl")
    (BASE_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
