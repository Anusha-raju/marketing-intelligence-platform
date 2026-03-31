from __future__ import annotations

from pathlib import Path

from marketing_intelligence.config import Paths
from marketing_intelligence.data_loader import load_raw_data, validate_raw_files
from marketing_intelligence.preprocessing import clean_raw_data
from marketing_intelligence.synthetic_marketing import generate_marketing_data, save_synthetic_outputs
from marketing_intelligence.features import build_customer_feature_mart
from marketing_intelligence.modeling import train_and_evaluate, save_model_artifacts
from marketing_intelligence.experimentation import evaluate_ab_test, save_ab_results
from marketing_intelligence.causal import estimate_iptw_effect, save_causal_results


def validate(project_root: Path) -> None:
    paths = Paths(project_root)
    validate_raw_files(paths.raw_dir)


def build_features(project_root: Path) -> None:
    paths = Paths(project_root)
    raw = clean_raw_data(load_raw_data(paths.raw_dir))
    syn = generate_marketing_data(raw["customers"], raw["orders"])
    save_synthetic_outputs(syn, paths.processed_dir)

    feature_mart = build_customer_feature_mart(
        customers=raw["customers"],
        orders=raw["orders"],
        order_items=raw["order_items"],
        order_payments=raw["order_payments"],
        order_reviews=raw["order_reviews"],
        marketing_touchpoints=syn.marketing_touchpoints,
        customer_sessions=syn.customer_sessions,
        experiment_assignments=syn.experiment_assignments,
    )
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    feature_mart.to_csv(paths.processed_dir / "customer_feature_mart.csv", index=False)


def train(project_root: Path) -> None:
    paths = Paths(project_root)
    feature_mart_path = paths.processed_dir / "customer_feature_mart.csv"
    if not feature_mart_path.exists():
        raise FileNotFoundError("Run build-features first")

    import pandas as pd

    feature_mart = pd.read_csv(feature_mart_path)
    conversion_model, conv_metrics, conv_features = train_and_evaluate(feature_mart, target="converted_30d")
    retention_model, ret_metrics, ret_features = train_and_evaluate(feature_mart, target="retained_180d")

    save_model_artifacts(
        conversion_model,
        conv_metrics,
        conv_features,
        paths.models_dir / "conversion_model.joblib",
        paths.models_dir / "conversion_metrics.json",
        paths.models_dir / "feature_columns.json",
    )
    save_model_artifacts(
        retention_model,
        ret_metrics,
        ret_features,
        paths.models_dir / "retention_model.joblib",
        paths.models_dir / "retention_metrics.json",
        paths.models_dir / "retention_feature_columns.json",
    )


def analyze(project_root: Path) -> None:
    paths = Paths(project_root)
    import pandas as pd

    feature_mart = pd.read_csv(paths.processed_dir / "customer_feature_mart.csv")
    ab = evaluate_ab_test(feature_mart)
    causal = estimate_iptw_effect(feature_mart)
    save_ab_results(ab, paths.processed_dir / "ab_test_results.json")
    save_causal_results(causal, paths.processed_dir / "causal_effects.json")
