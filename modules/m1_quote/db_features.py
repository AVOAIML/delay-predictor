"""DB-path feature templates for M1 (Connect-MaXXflow-Database retrain option).

Reads the tenant schema from the PostgreSQL READ REPLICA via the DAL (local docker
Postgres now; Azure PostgreSQL replica later — same DAL, config-swapped) and
reconstructs the gold-shaped training frames. Columns that don't exist in the
schema yet (region, industry, materialSpec, leadTimeDays) are simply not emitted —
the adaptive trainer trains on what's present and never synthesizes them.

Leakage-safe: per-contact / per-salesperson win-rates use only quotes closed
*before* each quote's created_at (Bayesian-smoothed)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from maxxflow_core import masterdata as MD
from maxxflow_data.engine import get_data_access
from maxxflow_data.masterdata_map import load_md_map

_SMOOTH = 3.0

_Q = """
SELECT id, stage_id, status_id, sales_order_id, grand_total, contact_id, sales_person_id,
       created_at, sales_order_created_at, lost_at
FROM quotations WHERE deleted_at IS NULL
"""
_LI = "SELECT id, quotation_id, product_id, quantity, unit_price, sales_price, line_amount FROM quotation_line_items"
_P = "SELECT id, name, unit_cost, sales_price FROM products WHERE deleted_at IS NULL"


def _label(q: pd.DataFrame, md) -> pd.DataFrame:
    win_stage = md.id("QUOTATION_STAGE", "SALES_ORDER")
    won_statuses = MD.resolve_ids(
        md, "QUOTATION_STATUS", MD.QUOTATION_WIN_STATUS_CODES
    )
    lost_statuses = MD.resolve_ids(
        md, "QUOTATION_STATUS", MD.QUOTATION_LOSS_STATUS_CODES
    )
    is_won = ((q["stage_id"] == win_stage) | q["status_id"].isin(won_statuses)
              | q["sales_order_id"].notna())
    is_lost = q["status_id"].isin(lost_statuses)
    q = q.copy()
    q["is_won"] = (is_won & ~is_lost)
    q["is_closed"] = q["is_won"] | is_lost
    q["won"] = np.where(q["is_won"], 1, np.where(is_lost, 0, np.nan))
    q["close_time"] = q["sales_order_created_at"].where(q["is_won"], q["lost_at"])
    return q


def _asof_rate(q: pd.DataFrame, entity: str) -> np.ndarray:
    """Bayesian-smoothed win-rate over each row's entity, from quotes closed earlier."""
    created = pd.to_datetime(q["created_at"]).to_numpy()
    close = pd.to_datetime(q["close_time"]).to_numpy()
    won = q["is_won"].to_numpy()
    closed = q["is_closed"].to_numpy()
    ent = q[entity].to_numpy()
    g = float(np.nanmean(q.loc[q["is_closed"], "won"])) if closed.any() else 0.5
    out = np.full(len(q), g)
    for i in range(len(q)):
        prior = closed & ~pd.isna(close) & (close < created[i]) & (ent == ent[i])
        n, w = int(prior.sum()), int(won[prior].sum())
        out[i] = (w + _SMOOTH * g) / (n + _SMOOTH)
    return out


# --- "Connect Database" honesty report -------------------------------------
# The gold training columns are DERIVED at train time from these RAW source
# tables; they are not physical columns. describe_sources() reports which source
# tables exist + have rows so the UI can show the truth (not "absent" features).
_SOURCE_SPEC = {
    "m1_quote_win": {
        "sources": {
            "quotations": ["grand_total", "stage_id", "status_id", "sales_order_id",
                           "contact_id", "sales_person_id", "created_at"],
            "quotation_line_items": ["quotation_id", "product_id", "quantity",
                                     "unit_price", "sales_price", "line_amount"],
            "products": ["id", "name"],
            "master_data": ["code", "category_id"],
        },
        "passthrough": ["grand_total"],
        "derived": ["total_quantity", "line_count", "n_products", "wtd_price_ratio",
                    "mean_price_ratio", "min_price_ratio", "avg_discount_pct",
                    "contact_win_rate", "salesrep_win_rate", "won"],
    },
    "m1_quote_price": {
        "sources": {
            "quotations": ["stage_id", "status_id", "sales_order_id", "contact_id", "created_at"],
            "quotation_line_items": ["quotation_id", "product_id", "quantity",
                                     "unit_price", "sales_price", "line_amount"],
            "products": ["id", "name"],
            "master_data": ["code", "category_id"],
        },
        "passthrough": [],
        "derived": ["productID", "unitPrice", "quantity", "price_ratio", "contact_win_rate"],
    },
    "m1_quote_line_win": {
        "sources": {
            "quotations": ["stage_id", "status_id", "sales_order_id", "contact_id",
                           "sales_person_id", "created_at"],
            "quotation_line_items": ["quotation_id", "product_id", "quantity",
                                     "unit_price", "sales_price", "line_amount"],
            "products": ["id", "name"],
            "master_data": ["code", "category_id"],
        },
        "passthrough": [],
        "derived": ["productID", "unitPrice", "quantity", "price_ratio",
                    "contact_win_rate", "salesrep_win_rate", "won"],
    },
}


def describe_sources(tenant: str, model_key: str) -> dict:
    """Honest report for the Configurator 'Connect Database' step: which RAW source
    tables the DB training path reads, whether they exist and have rows, and which
    gold features are DERIVED from them at training time (never physical columns)."""
    spec = _SOURCE_SPEC[model_key]
    da = get_data_access()
    cols = da.query("SELECT table_name, column_name FROM information_schema.columns "
                    "WHERE table_schema = current_schema()", tenant=tenant)
    by_table: dict[str, set] = {}
    for t, c in zip(cols["table_name"], cols["column_name"]):
        by_table.setdefault(str(t), set()).add(str(c).lower())
    try:
        schema = str(da.query("SELECT current_schema() AS s", tenant=tenant)["s"].iloc[0])
    except Exception:
        schema = None
    # If the tenant schema isn't provisioned, `SET search_path` falls back to `public`,
    # so current_schema() != the expected tenant schema -> the tenant isn't set up yet.
    from maxxflow_core.settings import get_settings
    expected = get_settings().tenant_schema(tenant)
    schema_exists = schema == expected
    sources, trainable = [], True
    for tbl, need in spec["sources"].items():
        present = tbl in by_table
        rows, missing = None, []
        if present:
            missing = [c for c in need if c.lower() not in by_table[tbl]]
            try:
                rows = int(da.query(f"SELECT count(*) AS n FROM {tbl}", tenant=tenant)["n"].iloc[0])
            except Exception:
                rows = None
        if (not present) or (rows == 0):
            trainable = False
        sources.append({"table": tbl, "present": present, "rows": rows,
                        "required_columns": need, "missing_columns": missing})
    return {"connected": True, "schema": schema, "expected_schema": expected,
            "schema_exists": schema_exists, "sources": sources,
            "passthrough_columns": spec["passthrough"], "derived_features": spec["derived"],
            "trainable": trainable}


def build_win_frame(tenant: str) -> pd.DataFrame:
    da = get_data_access()
    md = load_md_map(da, tenant)
    q = _label(da.query(_Q, tenant=tenant), md)
    li = da.query(_LI, tenant=tenant)
    li["ratio"] = pd.to_numeric(li["sales_price"], errors="coerce") / pd.to_numeric(li["unit_price"], errors="coerce").replace(0, np.nan)
    li["amt"] = pd.to_numeric(li["line_amount"], errors="coerce").fillna(0.0)
    grp = li.groupby("quotation_id")
    agg = pd.DataFrame({
        "total_quantity": grp["quantity"].apply(lambda s: float(pd.to_numeric(s, errors="coerce").sum())),
        "line_count": grp.size(),
        "n_products": grp["product_id"].nunique(),
        "mean_price_ratio": grp["ratio"].mean(),
        "min_price_ratio": grp["ratio"].min(),
        "wtd_price_ratio": grp.apply(lambda d: float(np.average(d["ratio"].fillna(1.0), weights=d["amt"]))
                                     if d["amt"].sum() > 0 else float(d["ratio"].mean()), include_groups=False),
    })
    q = q.merge(agg, left_on="id", right_index=True, how="left")
    q["avg_discount_pct"] = (1.0 - q["mean_price_ratio"].fillna(1.0)) * 100.0
    q = q.sort_values("created_at").reset_index(drop=True)
    q["contact_win_rate"] = _asof_rate(q, "contact_id")
    q["salesrep_win_rate"] = _asof_rate(q, "sales_person_id")
    out = q[q["is_closed"]].copy()
    out = out.rename(columns={"id": "quotationID"})
    cols = ["quotationID", "grand_total", "total_quantity", "line_count", "n_products",
            "wtd_price_ratio", "mean_price_ratio", "min_price_ratio", "avg_discount_pct",
            "contact_win_rate", "salesrep_win_rate", "won"]
    out["tenant"] = tenant
    return out[["tenant"] + cols]


def build_price_frame(tenant: str) -> pd.DataFrame:
    da = get_data_access()
    md = load_md_map(da, tenant)
    q = _label(da.query(_Q, tenant=tenant), md)
    won_ids = set(q.loc[q["is_won"], "id"])
    # per-contact as-of rate mapped onto lines via their quotation
    q = q.sort_values("created_at").reset_index(drop=True)
    q["contact_win_rate"] = _asof_rate(q, "contact_id")
    rate = dict(zip(q["id"], q["contact_win_rate"]))
    li = da.query(_LI, tenant=tenant)
    li = li[li["quotation_id"].isin(won_ids)].copy()
    prod = da.query(_P, tenant=tenant).rename(columns={"id": "product_id", "name": "productName"})
    li = li.merge(prod[["product_id", "productName"]], on="product_id", how="left")
    li["price_ratio"] = pd.to_numeric(li["sales_price"], errors="coerce") / pd.to_numeric(li["unit_price"], errors="coerce").replace(0, np.nan)
    li["contact_win_rate"] = li["quotation_id"].map(rate).fillna(0.5)
    out = li.rename(columns={"id": "quotationLineItemID", "product_id": "productID",
                             "unit_price": "unitPrice"})
    out["tenant"] = tenant
    cols = ["tenant", "quotationID", "quotationLineItemID", "productID", "productName",
            "quantity", "unitPrice", "contact_win_rate", "price_ratio"]
    return out[[c for c in cols if c in out.columns]].dropna(subset=["price_ratio"])


def build_line_frame(tenant: str) -> pd.DataFrame:
    """m1_quote_line_win training frame: one row per LINE of every CLOSED quote
    (won AND lost, unlike build_price_frame which is won-only) — the quote's won
    label is broadcast onto every one of its lines (see raw_ingest.build_line_frame
    for why: outcome is only recorded per-quotation, not per-line)."""
    da = get_data_access()
    md = load_md_map(da, tenant)
    q = _label(da.query(_Q, tenant=tenant), md)
    q = q.sort_values("created_at").reset_index(drop=True)
    q["contact_win_rate"] = _asof_rate(q, "contact_id")
    q["salesrep_win_rate"] = _asof_rate(q, "sales_person_id")
    closed = q[q["is_closed"]].rename(columns={"id": "quotationID"})

    li = da.query(_LI, tenant=tenant).rename(
        columns={"quotation_id": "quotationID", "product_id": "productID", "unit_price": "unitPrice"})
    li = li[li["quotationID"].isin(closed["quotationID"])].copy()
    li["price_ratio"] = (pd.to_numeric(li["sales_price"], errors="coerce")
                          / pd.to_numeric(li["unitPrice"], errors="coerce").replace(0, np.nan))

    out = li.merge(closed[["quotationID", "contact_win_rate", "salesrep_win_rate", "won"]],
                   on="quotationID", how="left")
    out["tenant"] = tenant
    cols = ["tenant", "quotationID", "productID", "quantity", "unitPrice", "price_ratio",
            "contact_win_rate", "salesrep_win_rate", "won"]
    return out[cols].dropna(subset=["price_ratio", "won"])
