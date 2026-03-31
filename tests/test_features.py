import pandas as pd

from marketing_intelligence.features import build_customer_feature_mart


def test_build_customer_feature_mart_basic():
    customers = pd.DataFrame([
        {"customer_id": "c1", "customer_unique_id": "u1", "customer_city": "sao paulo", "customer_state": "SP"}
    ])
    orders = pd.DataFrame([
        {"order_id": "o1", "customer_id": "c1", "order_purchase_timestamp": pd.Timestamp("2018-01-10"), "order_status": "delivered"},
        {"order_id": "o2", "customer_id": "c1", "order_purchase_timestamp": pd.Timestamp("2018-02-05"), "order_status": "delivered"},
    ])
    order_items = pd.DataFrame([
        {"order_id": "o1", "order_item_id": 1, "price": 100.0, "freight_value": 10.0},
        {"order_id": "o2", "order_item_id": 1, "price": 80.0, "freight_value": 8.0},
    ])
    order_payments = pd.DataFrame([
        {"order_id": "o1", "payment_value": 110.0, "payment_installments": 1},
        {"order_id": "o2", "payment_value": 88.0, "payment_installments": 2},
    ])
    order_reviews = pd.DataFrame([
        {"order_id": "o1", "review_score": 5},
        {"order_id": "o2", "review_score": 4},
    ])
    touchpoints = pd.DataFrame([
        {"customer_id": "c1", "session_id": "s1", "touch_timestamp": pd.Timestamp("2018-01-01"), "channel": "Email", "campaign_id": "cmp1", "campaign_type": "CRM", "impression_flag": 1, "click_flag": 1, "cost": 1.0, "device_type": "mobile", "geo_region": "SP"}
    ])
    sessions = pd.DataFrame([
        {"session_id": "s1", "customer_id": "c1", "session_start": pd.Timestamp("2018-01-01"), "landing_channel": "Email", "pages_viewed": 4, "session_duration_seconds": 120, "add_to_cart_flag": 1, "checkout_started_flag": 1, "purchase_flag": 1}
    ])
    experiments = pd.DataFrame([
        {"customer_id": "c1", "variant": "treatment", "eligibility_flag": 1, "exposed_flag": 1}
    ])

    mart = build_customer_feature_mart(customers, orders, order_items, order_payments, order_reviews, touchpoints, sessions, experiments)
    assert mart.shape[0] == 1
    assert int(mart.loc[0, "order_count"]) == 2
    assert int(mart.loc[0, "retained_180d"]) == 1
