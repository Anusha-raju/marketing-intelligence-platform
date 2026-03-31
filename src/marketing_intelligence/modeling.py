from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

RANDOM_STATE = 42


FEATURE_EXCLUDE = {
    "customer_id",
    "customer_unique_id",
    "first_purchase",
    "last_purchase",
    "first_touch",
    "last_touch",
    "converted_30d",
    "retained_180d",
}


def prepare_xy(feature_mart: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = feature_mart[[c for c in feature_mart.columns if c not in FEATURE_EXCLUDE]].copy()
    y = feature_mart[target].astype(int)
    return X, y


def make_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=2,
    )

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train_and_evaluate(feature_mart: pd.DataFrame, target: str) -> Tuple[Pipeline, Dict[str, float], list[str]]:
    X, y = prepare_xy(feature_mart, target)
    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify)

    pipeline = make_pipeline(X)
    pipeline.fit(X_train, y_train)
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probs)) if y_test.nunique() > 1 else np.nan,
        "pr_auc": float(average_precision_score(y_test, probs)) if y_test.nunique() > 1 else np.nan,
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "positive_rate": float(y.mean()),
    }
    return pipeline, metrics, X.columns.tolist()


def save_model_artifacts(model: Pipeline, metrics: Dict[str, float], feature_columns: list[str], model_path: Path, metrics_path: Path, feature_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    feature_path.write_text(json.dumps(feature_columns, indent=2))
