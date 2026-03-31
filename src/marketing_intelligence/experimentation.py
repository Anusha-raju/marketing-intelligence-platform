from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from statsmodels.stats.proportion import proportions_ztest


def evaluate_ab_test(feature_mart: pd.DataFrame) -> dict:
    eligible = feature_mart[feature_mart["eligibility_flag"] == 1].copy()
    grouped = eligible.groupby("variant").agg(
        customers=("customer_id", "nunique"),
        conversions=("converted_30d", "sum"),
        avg_revenue=("total_revenue", "mean"),
        retention_rate=("retained_180d", "mean"),
    )

    if {"control", "treatment"}.issubset(grouped.index):
        count = grouped.loc[["control", "treatment"], "conversions"].values
        nobs = grouped.loc[["control", "treatment"], "customers"].values
        stat, pval = proportions_ztest(count, nobs)
        lift = float((count[1] / nobs[1]) - (count[0] / nobs[0]))
    else:
        stat, pval, lift = 0.0, 1.0, 0.0

    result = {
        "summary": grouped.reset_index().to_dict(orient="records"),
        "z_stat": float(stat),
        "p_value": float(pval),
        "absolute_conversion_lift": lift,
    }
    return result


def save_ab_results(results: dict, path: Path) -> None:
    path.write_text(json.dumps(results, indent=2))
