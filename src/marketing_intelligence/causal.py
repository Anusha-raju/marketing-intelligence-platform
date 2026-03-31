from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
EXCLUDE = {
    "customer_id",
    "customer_unique_id",
    "first_purchase",
    "last_purchase",
    "first_touch",
    "last_touch",
    "converted_30d",
    "retained_180d",
    "is_treatment",
    "variant",
}


def estimate_iptw_effect(feature_mart: pd.DataFrame) -> dict:
    df = feature_mart[feature_mart["eligibility_flag"] == 1].copy()
    y = df["is_treatment"].astype(int)
    X = df[[c for c in df.columns if c not in EXCLUDE]].copy()

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    prep = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    model = Pipeline([("prep", prep), ("lr", LogisticRegression(max_iter=5000, solver="lbfgs"))])
    model.fit(X, y)
    propensity = model.predict_proba(X)[:, 1]
    propensity = np.clip(propensity, 0.05, 0.95)

    treatment = df["is_treatment"].values
    outcome = df["converted_30d"].values

    weights = treatment / propensity + (1 - treatment) / (1 - propensity)
    treated_mean = np.sum(weights * treatment * outcome) / np.sum(weights * treatment)
    control_mean = np.sum(weights * (1 - treatment) * outcome) / np.sum(weights * (1 - treatment))
    naive = df.groupby("is_treatment")["converted_30d"].mean().to_dict()

    return {
        "naive_control_rate": float(naive.get(0, 0.0)),
        "naive_treatment_rate": float(naive.get(1, 0.0)),
        "naive_difference": float(naive.get(1, 0.0) - naive.get(0, 0.0)),
        "iptw_treatment_rate": float(treated_mean),
        "iptw_control_rate": float(control_mean),
        "iptw_ate": float(treated_mean - control_mean),
    }


def save_causal_results(results: dict, path: Path) -> None:
    path.write_text(json.dumps(results, indent=2))
