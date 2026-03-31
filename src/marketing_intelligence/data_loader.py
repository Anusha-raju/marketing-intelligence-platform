from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

REQUIRED_FILES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "translation": "product_category_name_translation.csv",
    "mql": "olist_marketing_qualified_leads_dataset.csv",
    "closed_deals": "olist_closed_deals_dataset.csv",
}


def validate_raw_files(data_dir: Path) -> None:
    missing = [filename for filename in REQUIRED_FILES.values() if not (data_dir / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw files: {missing}")


def load_raw_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    validate_raw_files(data_dir)
    data = {}
    for key, filename in REQUIRED_FILES.items():
        df = pd.read_csv(data_dir / filename)
        data[key] = df
    return data
