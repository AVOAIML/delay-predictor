"""M1 Smart Quote Optimizer — feature engineering (plan §4 M1, §1a).

ONE implementation, used by BOTH the synthetic gate and training/scoring, fed the
raw tables the DAL returns from Postgres (so feature logic is identical to prod).

Leakage-safe: per-contact and per-salesperson prior win-rates are computed only
from quotes *closed before* this quote's ``created_at`` (no target leakage).
Margins use ``decimal.Decimal`` (a float can flip a Win/Loss at the boundary) and
recompute from line items — the cached ``line_amount`` is not trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from maxxflow_core import masterdata as MD
from maxxflow_core.clock import Clock
from maxxflow_core.money import D, gross_margin_fraction

# Model input columns (order matters for the served signature).
FEATURE_COLUMNS = [
    "gross_margin",
    "log_value",
    "total_quantity",
    "days_to_expiration",
    "contact_prior_winrate",
    "salesperson_prior_winrate",
    "product_type_code",
]
CATEGORICAL = ["product_type_code"]
_SMOOTH = 3.0  # Bayesian smoothing for as-of prior win-rates


def _product_type_reverse(md) -> dict:
    return {md.id("PRODUCT_TYPE", code): code for code in MD.category_codes("PRODUCT_TYPE")}


def build_features(tables: dict[str, pd.DataFrame], md, clock: Clock) -> pd.DataFrame:
    q = tables["quotations"].copy()
    li = tables["quotation_line_items"]
    prod = tables["products"][["id", "unit_cost", "product_type_id"]].rename(
        columns={"id": "product_id", "unit_cost": "product_unit_cost"}
    )

    # --- labels from MasterData UUIDs (plan §1a row 2) ---
    win_stage = md.id("QUOTATION_STAGE", "SALES_ORDER")
    won_statuses = MD.resolve_ids(md, "QUOTATION_STATUS", MD.QUOTATION_WIN_STATUS_CODES)
    lost_statuses = MD.resolve_ids(md, "QUOTATION_STATUS", MD.QUOTATION_LOSS_STATUS_CODES)
    is_won = ((q["stage_id"] == win_stage) | q["status_id"].isin(won_statuses)
              | q["sales_order_id"].notna())
    is_lost = q["status_id"].isin(lost_statuses)
    q["is_won"] = is_won & ~is_lost
    q["is_closed"] = q["is_won"] | is_lost
    q["label"] = np.where(q["is_won"], 1.0, np.where(is_lost, 0.0, np.nan))
    q["close_time"] = q["sales_order_created_at"].where(q["is_won"], q["lost_at"])

    # --- per-line margin (Decimal) -> weighted avg per quotation ---
    lij = li.merge(prod, on="product_id", how="left")
    margins, weights = [], []
    for sp, uc, amt in zip(lij["sales_price"], lij["product_unit_cost"], lij["line_amount"]):
        mf = gross_margin_fraction(sp, uc)
        margins.append(float(mf) if mf is not None else 0.0)
        weights.append(float(D(amt)))
    lij = lij.assign(_m=margins, _w=weights)
    grp = lij.groupby("quotation_id")

    def _weighted_margin(d: pd.DataFrame) -> float:
        # A line's product_unit_cost is NaN when its product row didn't join
        # (e.g. soft-deleted: dal.py filters products by deleted_at but not
        # quotation_line_items), which makes _m NaN. Drop just that line so it
        # can't NaN out the whole quote's weighted average.
        valid = d[d["_m"].notna()]
        if valid.empty:
            return np.nan
        return np.average(valid["_m"], weights=valid["_w"]) if valid["_w"].sum() > 0 else valid["_m"].mean()

    wmargin = grp.apply(_weighted_margin, include_groups=False)
    qty = grp["quantity"].apply(lambda s: float(sum(D(x) for x in s)))
    feat = pd.DataFrame({"gross_margin": wmargin, "total_quantity": qty})

    q = q.merge(feat, left_on="id", right_index=True, how="left")
    q["gross_margin"] = q["gross_margin"].fillna(0.0)
    q["total_quantity"] = q["total_quantity"].fillna(0.0)
    q["log_value"] = np.log1p(q["grand_total"].astype(float).clip(lower=0))

    # days-to-expiration = expiration - created (tz-invariant: naive subtraction)
    exp = pd.to_datetime(q["expiration_date"])
    cre = pd.to_datetime(q["created_at"])
    # Same-day expiry stores expiration_date as a bare Date but created_at keeps
    # its time-of-day, so exp-cre goes negative even for a legitimate quote (the
    # app validates expiration against "today", not created_at). Floor at 0 —
    # same sentinel already used for a missing expiration_date below.
    q["days_to_expiration"] = ((exp - cre).dt.total_seconds() / 86400.0).clip(lower=0.0)
    q["days_to_expiration"] = q["days_to_expiration"].fillna(0.0)

    # product type code (majority line per quote)
    rev = _product_type_reverse(md)
    lij["ptype"] = lij["product_type_id"].map(rev)
    ptype = lij.groupby("quotation_id")["ptype"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else "FINISHED_GOOD")
    q = q.merge(ptype.rename("product_type_code"), left_on="id", right_index=True, how="left")
    q["product_type_code"] = q["product_type_code"].fillna("FINISHED_GOOD").astype("category")

    # --- as-of prior win-rates (leakage-safe) ---
    # Point-in-time join instead of an O(n^2) per-quote scan: cumulative wins/counts
    # per contact and per seller, computed once in close_time order, then matched to
    # each quote via merge_asof(direction="backward", allow_exact_matches=False) —
    # the latest same-contact/seller close strictly before this quote's created_at.
    q = q.sort_values("created_at").reset_index(drop=True)
    q["created_at"] = pd.to_datetime(q["created_at"])
    q["close_time"] = pd.to_datetime(q["close_time"])

    global_rate = float(np.nanmean(q.loc[q["is_closed"], "label"])) if q["is_closed"].any() else 0.5

    closed_df = (
        q.loc[q["is_closed"] & q["close_time"].notna(), ["contact_id", "sales_person_id", "close_time", "is_won"]]
        .sort_values("close_time")
    )
    closed_df["contact_cum_wins"] = closed_df.groupby("contact_id")["is_won"].cumsum()
    closed_df["contact_cum_count"] = closed_df.groupby("contact_id").cumcount() + 1
    closed_df["seller_cum_wins"] = closed_df.groupby("sales_person_id")["is_won"].cumsum()
    closed_df["seller_cum_count"] = closed_df.groupby("sales_person_id").cumcount() + 1

    c_match = pd.merge_asof(
        q[["created_at", "contact_id"]],
        closed_df[["close_time", "contact_id", "contact_cum_wins", "contact_cum_count"]],
        left_on="created_at", right_on="close_time", by="contact_id",
        direction="backward", allow_exact_matches=False,
    )
    s_match = pd.merge_asof(
        q[["created_at", "sales_person_id"]],
        closed_df[["close_time", "sales_person_id", "seller_cum_wins", "seller_cum_count"]],
        left_on="created_at", right_on="close_time", by="sales_person_id",
        direction="backward", allow_exact_matches=False,
    )
    nc = c_match["contact_cum_count"].fillna(0.0).to_numpy()
    wc = c_match["contact_cum_wins"].fillna(0.0).to_numpy()
    ns = s_match["seller_cum_count"].fillna(0.0).to_numpy()
    ws = s_match["seller_cum_wins"].fillna(0.0).to_numpy()

    q["contact_prior_winrate"] = (wc + _SMOOTH * global_rate) / (nc + _SMOOTH)
    q["salesperson_prior_winrate"] = (ws + _SMOOTH * global_rate) / (ns + _SMOOTH)
    q["n_comparable"] = nc.astype(int)

    return q[["id", "created_at", "is_closed", "label", "n_comparable", "grand_total"] + FEATURE_COLUMNS]


def training_frame(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    closed = features[features["is_closed"]].copy()
    return closed[FEATURE_COLUMNS], closed["label"].astype(int)
