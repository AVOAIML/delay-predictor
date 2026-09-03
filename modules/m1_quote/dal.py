"""M1 feature SQL — the byte-identical-to-prod reads (plan §1, §1a).

These SELECTs run against the local Postgres (or the Azure read replica in
Phase 2) through the DAL, which sets ``search_path`` and forbids ``tenant_id`` /
``rop_status``. The returned column sets match ``schema_def`` so the generator's
rows and these reads are interchangeable (the contract test enforces this).
"""

from __future__ import annotations

import pandas as pd

from maxxflow_data.engine import get_data_access
from maxxflow_data.masterdata_map import load_md_map

_Q_QUOTATIONS = """
SELECT id, quotation_id, sales_order_id, contact_id, organisation_id, stage_id, status_id,
       quote_type, expiration_date, payment_terms_id, total_amount, tax_percentage, tax_amount,
       grand_total, sales_person_id, sent_at, lost_at, sales_order_created_at, created_at
FROM quotations
WHERE deleted_at IS NULL
"""
_Q_LINES = """
SELECT id, quotation_id, product_id, quantity, unit_price, sales_price, line_amount, sort_order
FROM quotation_line_items
"""
_Q_PRODUCTS = """
SELECT id, sku, product_type_id, sales_price, unit_cost
FROM products
WHERE deleted_at IS NULL
"""


def read_quote_tables(tenant: str = "demo") -> tuple[dict[str, pd.DataFrame], object]:
    da = get_data_access()
    md = load_md_map(da, tenant)
    tables = {
        "quotations": da.query(_Q_QUOTATIONS, tenant=tenant),
        "quotation_line_items": da.query(_Q_LINES, tenant=tenant),
        "products": da.query(_Q_PRODUCTS, tenant=tenant),
    }
    return tables, md
