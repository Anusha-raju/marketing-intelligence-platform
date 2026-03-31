from __future__ import annotations

import numpy as np
import pandas as pd

from marketing_intelligence.config import RETENTION_REPEAT_PURCHASE_DAYS, TARGET_CONVERSION_WINDOW_DAYS


def build_customer_feature_mart(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    order_payments: pd.DataFrame,
    order_reviews: pd.DataFrame,
    marketing_touchpoints: pd.DataFrame,
    customer_sessions: pd.DataFrame,
    experiment_assignments: pd.DataFrame,
) -> pd.DataFrame:
    orders_small = orders[["order_id", "customer_id", "order_purchase_timestamp", "order_status"]].copy()

    order_rev = order_items.groupby("order_id", as_index=False).agg(
        item_price=("price", "sum"), freight_value=("freight_value", "sum"), item_count=("order_item_id", "count")
    )
    payments = order_payments.groupby("order_id", as_index=False).agg(
        payment_value=("payment_value", "sum"), payment_installments=("payment_installments", "max")
    )
    reviews = order_reviews.groupby("order_id", as_index=False).agg(review_score=("review_score", "mean"))

    order_fact = (
        orders_small.merge(order_rev, on="order_id", how="left")
        .merge(payments, on="order_id", how="left")
        .merge(reviews, on="order_id", how="left")
    )

    cust_orders = order_fact.groupby("customer_id", as_index=False).agg(
        first_purchase=("order_purchase_timestamp", "min"),
        last_purchase=("order_purchase_timestamp", "max"),
        order_count=("order_id", "nunique"),
        total_revenue=("payment_value", "sum"),
        avg_order_value=("payment_value", "mean"),
        avg_review_score=("review_score", "mean"),
        avg_items_per_order=("item_count", "mean"),
        avg_freight_value=("freight_value", "mean"),
    )

    max_date = orders_small["order_purchase_timestamp"].max()

    touches = marketing_touchpoints.copy()
    touches["touch_timestamp"] = pd.to_datetime(touches["touch_timestamp"])
    sessions = customer_sessions.copy()
    sessions["session_start"] = pd.to_datetime(sessions["session_start"])
    experiment_assignments = experiment_assignments.copy()

    touch_agg = touches.groupby("customer_id", as_index=False).agg(
        first_touch=("touch_timestamp", "min"),
        last_touch=("touch_timestamp", "max"),
        num_touches=("channel", "count"),
        num_clicks=("click_flag", "sum"),
        total_marketing_cost=("cost", "sum"),
        unique_channels=("channel", "nunique"),
    )

    channel_pivot = (
        touches.pivot_table(index="customer_id", columns="channel", values="impression_flag", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    channel_pivot.columns = ["customer_id"] + [f"impressions_{str(c).lower().replace(' ', '_')}" for c in channel_pivot.columns[1:]]

    click_pivot = (
        touches.pivot_table(index="customer_id", columns="channel", values="click_flag", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    click_pivot.columns = ["customer_id"] + [f"clicks_{str(c).lower().replace(' ', '_')}" for c in click_pivot.columns[1:]]

    session_agg = sessions.groupby("customer_id", as_index=False).agg(
        sessions=("session_id", "nunique"),
        avg_pages_viewed=("pages_viewed", "mean"),
        avg_session_duration=("session_duration_seconds", "mean"),
        add_to_cart_sessions=("add_to_cart_flag", "sum"),
        checkout_sessions=("checkout_started_flag", "sum"),
        purchase_sessions=("purchase_flag", "sum"),
    )

    base = customers[["customer_id", "customer_unique_id", "customer_city", "customer_state"]].copy()
    feature_mart = (
        base.merge(cust_orders, on="customer_id", how="left")
        .merge(touch_agg, on="customer_id", how="left")
        .merge(channel_pivot, on="customer_id", how="left")
        .merge(click_pivot, on="customer_id", how="left")
        .merge(session_agg, on="customer_id", how="left")
        .merge(experiment_assignments[["customer_id", "variant", "eligibility_flag", "exposed_flag"]], on="customer_id", how="left")
    )

    feature_mart["days_since_last_purchase"] = (max_date - feature_mart["last_purchase"]).dt.days
    feature_mart["days_since_last_touch"] = (max_date - feature_mart["last_touch"]).dt.days
    feature_mart["days_from_first_touch_to_purchase"] = (feature_mart["first_purchase"] - feature_mart["first_touch"]).dt.days
    feature_mart["touch_to_click_rate"] = feature_mart["num_clicks"] / feature_mart["num_touches"].replace(0, np.nan)
    feature_mart["cart_rate"] = feature_mart["add_to_cart_sessions"] / feature_mart["sessions"].replace(0, np.nan)
    feature_mart["checkout_rate"] = feature_mart["checkout_sessions"] / feature_mart["sessions"].replace(0, np.nan)
    feature_mart["purchase_session_rate"] = feature_mart["purchase_sessions"] / feature_mart["sessions"].replace(0, np.nan)

    feature_mart["converted_30d"] = (
        feature_mart["days_from_first_touch_to_purchase"].fillna(9999) <= TARGET_CONVERSION_WINDOW_DAYS
    ).astype(int)

    order_ranks = order_fact.sort_values(["customer_id", "order_purchase_timestamp"])
    order_ranks["next_purchase"] = order_ranks.groupby("customer_id")["order_purchase_timestamp"].shift(-1)
    repeat_gap = order_ranks.groupby("customer_id", as_index=False).agg(min_days_to_next=("next_purchase", lambda s: np.nan))
    temp = order_ranks.copy()
    temp["days_to_next"] = (temp["next_purchase"] - temp["order_purchase_timestamp"]).dt.days
    repeat_gap = temp.groupby("customer_id", as_index=False).agg(min_days_to_next=("days_to_next", "min"))
    feature_mart = feature_mart.merge(repeat_gap, on="customer_id", how="left")
    feature_mart["retained_180d"] = (feature_mart["min_days_to_next"].fillna(9999) <= RETENTION_REPEAT_PURCHASE_DAYS).astype(int)

    feature_mart["variant"] = feature_mart["variant"].fillna("control")
    feature_mart["is_treatment"] = (feature_mart["variant"] == "treatment").astype(int)

    numeric_cols = feature_mart.select_dtypes(include=[np.number]).columns
    feature_mart[numeric_cols] = feature_mart[numeric_cols].fillna(0)

    for c in ["customer_city", "customer_state"]:
        feature_mart[c] = feature_mart[c].fillna("unknown")

    return feature_mart
