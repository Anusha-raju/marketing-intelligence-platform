from __future__ import annotations

from typing import Dict

import pandas as pd


def _to_datetime(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    return out


def clean_raw_data(raw: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    cleaned = raw.copy()
    cleaned["orders"] = _to_datetime(
        cleaned["orders"],
        [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    cleaned["order_reviews"] = _to_datetime(cleaned["order_reviews"], ["review_creation_date", "review_answer_timestamp"])
    cleaned["mql"] = _to_datetime(cleaned["mql"], ["first_contact_date"])
    cleaned["closed_deals"] = _to_datetime(cleaned["closed_deals"], ["won_date"])
    return cleaned
