from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

CHANNELS = [
    "Paid Search",
    "Paid Social",
    "Email",
    "Affiliate",
    "Display",
    "Organic Search",
    "Direct",
]

CAMPAIGN_TYPES = ["Prospecting", "Retargeting", "CRM", "Brand", "Promo"]
DEVICE_TYPES = ["mobile", "desktop", "tablet"]
REGIONS = ["SP", "RJ", "MG", "BA", "PR", "RS", "SC", "Other"]


@dataclass
class SyntheticOutputs:
    campaigns: pd.DataFrame
    marketing_touchpoints: pd.DataFrame
    customer_sessions: pd.DataFrame
    experiment_assignments: pd.DataFrame


def _channel_probabilities(state: str | float) -> np.ndarray:
    if state in {"SP", "RJ", "MG"}:
        probs = np.array([0.20, 0.18, 0.15, 0.08, 0.10, 0.17, 0.12])
    else:
        probs = np.array([0.16, 0.14, 0.12, 0.10, 0.10, 0.20, 0.18])
    return probs / probs.sum()


def _make_campaigns(rng: np.random.Generator, n: int = 28) -> pd.DataFrame:
    rows = []
    for i in range(n):
        channel = CHANNELS[i % len(CHANNELS)]
        start = pd.Timestamp("2017-01-01") + pd.Timedelta(days=int(rng.integers(0, 700)))
        end = start + pd.Timedelta(days=int(rng.integers(21, 90)))
        rows.append(
            {
                "campaign_id": f"cmp_{i+1:03d}",
                "campaign_name": f"{channel.lower().replace(' ', '_')}_{i+1:03d}",
                "channel": channel,
                "campaign_type": rng.choice(CAMPAIGN_TYPES),
                "audience_segment": rng.choice(["new_users", "high_intent", "discount_seekers", "repeat_buyers"]),
                "start_date": start,
                "end_date": end,
                "creative_type": rng.choice(["video", "static", "search_text", "html_email"]),
                "target_objective": rng.choice(["acquisition", "conversion", "reactivation", "retention"]),
                "daily_budget": int(rng.integers(400, 6000)),
            }
        )
    return pd.DataFrame(rows)


def generate_marketing_data(customers: pd.DataFrame, orders: pd.DataFrame, random_state: int = 42) -> SyntheticOutputs:
    rng = np.random.default_rng(random_state)
    campaigns = _make_campaigns(rng)

    customer_order_base = (
        orders.groupby("customer_id")
        .agg(
            first_purchase=("order_purchase_timestamp", "min"),
            last_purchase=("order_purchase_timestamp", "max"),
            order_count=("order_id", "nunique"),
        )
        .reset_index()
    )
    customers_base = customers.merge(customer_order_base, on="customer_id", how="left")
    max_purchase = orders["order_purchase_timestamp"].max()
    global_end = max_purchase if pd.notnull(max_purchase) else pd.Timestamp("2018-08-31")

    touch_rows = []
    session_rows = []
    experiment_rows = []

    for row in customers_base.itertuples(index=False):
        order_count = 0 if pd.isna(row.order_count) else int(row.order_count)
        state = getattr(row, "customer_state", "Other")
        num_touches = int(np.clip(rng.poisson(2 + order_count), 1, 12))
        channel_probs = _channel_probabilities(state)
        conversion_anchor = row.first_purchase if pd.notnull(row.first_purchase) else global_end - pd.Timedelta(days=int(rng.integers(15, 180)))

        for t in range(num_touches):
            channel = rng.choice(CHANNELS, p=channel_probs)
            eligible_campaigns = campaigns[campaigns["channel"] == channel]
            campaign = eligible_campaigns.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
            lag_days = int(rng.integers(1, 120))
            ts = conversion_anchor - pd.Timedelta(days=lag_days) + pd.Timedelta(hours=int(rng.integers(0, 24)))
            impression = 1
            ctr_base = {
                "Paid Search": 0.20,
                "Paid Social": 0.10,
                "Email": 0.17,
                "Affiliate": 0.09,
                "Display": 0.04,
                "Organic Search": 0.28,
                "Direct": 0.35,
            }[channel]
            click = int(rng.random() < min(0.9, ctr_base + 0.02 * order_count))
            cost = round(float(rng.uniform(0.01, 6.00) * (1 + (channel in {"Paid Search", "Paid Social", "Display"}))), 2)
            session_id = f"sess_{row.customer_id}_{t+1}"
            pages = int(np.clip(rng.normal(3 + 1.2 * click + 0.8 * order_count, 1.6), 1, 20))
            duration = int(np.clip(rng.normal(90 + 35 * click + 25 * order_count, 40), 15, 1800))
            add_to_cart = int(rng.random() < (0.15 + 0.18 * click + 0.04 * min(order_count, 3)))
            purchase = int(pd.notnull(row.first_purchase) and (conversion_anchor - ts).days <= 30 and click == 1 and rng.random() < 0.55)
            checkout = int(add_to_cart == 1 and rng.random() < 0.7)

            touch_rows.append(
                {
                    "customer_id": row.customer_id,
                    "session_id": session_id,
                    "touch_timestamp": ts,
                    "channel": channel,
                    "campaign_id": campaign.campaign_id,
                    "campaign_type": campaign.campaign_type,
                    "impression_flag": impression,
                    "click_flag": click,
                    "cost": cost,
                    "device_type": rng.choice(DEVICE_TYPES, p=[0.62, 0.32, 0.06]),
                    "geo_region": state if pd.notnull(state) else "Other",
                }
            )
            session_rows.append(
                {
                    "session_id": session_id,
                    "customer_id": row.customer_id,
                    "session_start": ts,
                    "landing_channel": channel,
                    "pages_viewed": pages,
                    "session_duration_seconds": duration,
                    "add_to_cart_flag": add_to_cart,
                    "checkout_started_flag": checkout,
                    "purchase_flag": purchase,
                }
            )

        experiment_rows.append(
            {
                "customer_id": row.customer_id,
                "experiment_id": "exp_email_discount_001",
                "variant": rng.choice(["control", "treatment"], p=[0.5, 0.5]),
                "assignment_date": global_end - pd.Timedelta(days=90),
                "eligibility_flag": int(order_count <= 2),
                "exposed_flag": int(rng.random() < 0.88),
            }
        )

    marketing_touchpoints = pd.DataFrame(touch_rows).sort_values(["customer_id", "touch_timestamp"])
    customer_sessions = pd.DataFrame(session_rows).sort_values(["customer_id", "session_start"])
    experiment_assignments = pd.DataFrame(experiment_rows)

    return SyntheticOutputs(
        campaigns=campaigns,
        marketing_touchpoints=marketing_touchpoints,
        customer_sessions=customer_sessions,
        experiment_assignments=experiment_assignments,
    )


def save_synthetic_outputs(outputs: SyntheticOutputs, processed_dir) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs.campaigns.to_csv(processed_dir / "campaigns.csv", index=False)
    outputs.marketing_touchpoints.to_csv(processed_dir / "marketing_touchpoints.csv", index=False)
    outputs.customer_sessions.to_csv(processed_dir / "customer_sessions.csv", index=False)
    outputs.experiment_assignments.to_csv(processed_dir / "experiment_assignments.csv", index=False)

    metadata = {
        "channels": CHANNELS,
        "campaign_types": CAMPAIGN_TYPES,
    }
    (processed_dir / "synthetic_metadata.json").write_text(json.dumps(metadata, indent=2))
