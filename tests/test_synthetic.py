import pandas as pd

from marketing_intelligence.synthetic_marketing import generate_marketing_data


def test_generate_marketing_data_outputs():
    customers = pd.DataFrame([
        {"customer_id": "c1", "customer_state": "SP"},
        {"customer_id": "c2", "customer_state": "RJ"},
    ])
    orders = pd.DataFrame([
        {"order_id": "o1", "customer_id": "c1", "order_purchase_timestamp": pd.Timestamp("2018-01-10")},
        {"order_id": "o2", "customer_id": "c2", "order_purchase_timestamp": pd.Timestamp("2018-02-10")},
    ])
    out = generate_marketing_data(customers, orders)
    assert not out.campaigns.empty
    assert not out.marketing_touchpoints.empty
    assert not out.customer_sessions.empty
    assert not out.experiment_assignments.empty
