"""M1 scoring + writeback (plan §4 M1 serve, §8 audit).

Advisory win-probability + price band written to ``Quotation.customElements``
(JSON, no migration) so the grid reads a cached value with no blocking AI call.
Low-confidence and hidden (<5 comparable) scores are logged to the audit table,
not silently dropped (BRD audit requirement).
"""

from __future__ import annotations

import json

import pandas as pd

from maxxflow_core.clock import get_clock
from maxxflow_core.errors import get_logger
from maxxflow_core.jsonutil import json_default
from maxxflow_core.money import D
from maxxflow_mlops.naming import registered_model_name
from m1_quote.features import FEATURE_COLUMNS, build_features

log = get_logger("m1_quote.score")
MODULE = "m1_quote"
MODULE_DISPLAY_NAME = "Smart Quote Optimizer"  # human-readable label for audit_logs.module


def base_price_by_quote(tables: dict) -> dict:
    """Representative base Product.salesPrice per quote = the largest line's product."""
    li = tables["quotation_line_items"]
    prod = tables["products"][["id", "sales_price"]].rename(
        columns={"id": "product_id", "sales_price": "base_price"})
    j = li.merge(prod, on="product_id", how="left")
    idx = j.groupby("quotation_id")["line_amount"].idxmax()
    top = j.loc[idx, ["quotation_id", "base_price"]]
    return dict(zip(top["quotation_id"], top["base_price"].astype(float)))


def build_scoring_records(feature_frame: pd.DataFrame, tables: dict, *, only_open: bool = True) -> pd.DataFrame:
    df = feature_frame[~feature_frame["is_closed"]] if only_open else feature_frame
    bp = base_price_by_quote(tables)
    recs = df[["id"] + FEATURE_COLUMNS + ["n_comparable"]].copy()
    # match the served signature: string category + float int-feature
    recs["product_type_code"] = recs["product_type_code"].astype(str)
    recs["n_comparable"] = recs["n_comparable"].astype(float)
    recs["base_price"] = recs["id"].map(bp)
    return recs


def score_records(model, records: pd.DataFrame) -> pd.DataFrame:
    feats = records.drop(columns=["id"])
    preds = model.predict(feats)
    preds.insert(0, "id", records["id"].to_numpy())
    return preds


def _advisory_payload(row, model_version) -> dict:
    clock = get_clock()
    return {"ai_quote": {
        "win_probability_pct": row["win_probability_pct"],
        "recommended_price_low": row["recommended_price_low"],
        "recommended_price_high": row["recommended_price_high"],
        "low_confidence": row["low_confidence"],
        "display": not row["hidden"],
        "price_clamped": row["price_clamped"],
        "model_version": model_version,
        "scored_at": clock.as_of().isoformat(),
    }}


def _bulk_update_advisory(da, tenant: str, updates: list[tuple[str, str]]) -> None:
    """One UPDATE .. FROM (VALUES ..) round trip for every quote instead of one
    UPDATE per quote — each row still merges its own distinct payload via the
    VALUES join, so per-quote semantics are unchanged."""
    if not updates:
        return
    values_sql = ", ".join(f"(:id_{i}, :payload_{i})" for i in range(len(updates)))
    params: dict[str, str] = {}
    for i, (qid, payload) in enumerate(updates):
        params[f"id_{i}"] = qid
        params[f"payload_{i}"] = payload
    da.execute(
        "UPDATE quotations AS q "
        "SET custom_elements = COALESCE(q.custom_elements,'{}'::jsonb) || v.payload::jsonb "
        f"FROM (VALUES {values_sql}) AS v(id, payload) "
        "WHERE q.id = v.id::uuid",
        params, tenant=tenant,
    )


def _bulk_insert_audit(da, tenant: str, rows: list[dict]) -> None:
    """One multi-row INSERT instead of one INSERT per low-confidence/hidden quote."""
    if not rows:
        return
    values_sql = ", ".join(
        f"(:module_name, 'score', 'Quotation', :id_{i}, CAST(:m_{i} AS jsonb), now())"
        for i in range(len(rows))
    )
    params: dict[str, str] = {"module_name": MODULE_DISPLAY_NAME}
    for i, r in enumerate(rows):
        params[f"id_{i}"] = r["id"]
        params[f"m_{i}"] = r["metadata"]
    da.execute(
        "INSERT INTO audit_logs (module, action, entity_type, entity_id, metadata, timestamp) "
        f"VALUES {values_sql}",
        params, tenant=tenant,
    )


def score(tenant: str = "demo") -> int:
    from maxxflow_data.engine import get_data_access
    from maxxflow_mlops.registry import MLflowRegistry
    from m1_quote.dal import read_quote_tables

    tables, md = read_quote_tables(tenant)
    feats = build_features(tables, md, get_clock())
    records = build_scoring_records(feats, tables, only_open=True)
    if records.empty:
        return 0

    reg = MLflowRegistry()
    name = registered_model_name(tenant, MODULE)
    model = reg.load_champion(name=name)
    version = reg.get_alias_version(name=name, alias="champion")
    preds = score_records(model, records)

    updates: list[tuple[str, str]] = []
    audit_rows: list[dict] = []
    for _, row in preds.iterrows():
        payload = _advisory_payload(row, version)
        updates.append((row["id"], json.dumps(payload, default=json_default)))
        if row["low_confidence"] or row["hidden"]:
            audit_rows.append({
                "id": row["id"],
                "metadata": json.dumps(
                    {"low_confidence": bool(row["low_confidence"]), "hidden": bool(row["hidden"]),
                     "win_probability_pct": row["win_probability_pct"]}, default=json_default),
            })

    da = get_data_access()
    _bulk_update_advisory(da, tenant, updates)
    _bulk_insert_audit(da, tenant, audit_rows)

    n = len(updates)
    log.info("scored %d open quotes for tenant=%s (model v%s)", n, tenant, version)
    return n
