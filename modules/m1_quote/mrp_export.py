"""Read the raw csv_line_win training export straight out of a tenant's MRP schema.

One SQL statement, one place. Both callers use it:

  * ``db/export_m1_line_win.py``  — CLI, writes a CSV to the lake
  * ``POST .../read_data_from_mrp`` — the Connect-MaXXflow-Database button

The statement itself lives in ``db/m1_line_win_export.sql`` rather than in a
string here, so it stays readable, diffable and runnable in psql by hand. That
folder is COPY'd into the image (see docker/Dockerfile's ``src`` stage), so the
service can read it at runtime.

WHY THIS RETURNS THE *RAW* EXPORT SHAPE AND NOT GOLD FEATURES
-------------------------------------------------------------
``db_features.build_line_frame`` already reads Postgres and emits gold features
directly, and it deliberately drops the columns the schema has no source for
(industry, leadTimeDays, and until recently region and materialSpec). That is the
right call for a headless retrain, but it means the operator never sees the data.
This path instead reconstructs the RAW combined-export shape — the same CSV a
client would upload — so it can go through ``raw_ingest.build_line_frame`` and be
previewed, validated and diffed against a file upload column for column. The MRP
read becomes just another upload, which is why training needs no new code path.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# db/ sits at the repo root, two levels up from modules/m1_quote/
SQL_PATH = Path(__file__).resolve().parents[2] / "db" / "m1_line_win_export.sql"
# the file ends with a long explanatory comment block, fenced off by a box-drawing
# rule. Everything after it is prose for a human reading the file in psql.
_FENCE = "-- " + "─" * 74

# What the MRP schema genuinely cannot supply, and what we do instead. Surfaced in
# the API response so nobody has to read this module to find out.
UNSOURCED = {
    "industry": "no industry/vertical column exists in any of the 69 MRP models — "
                "supplied by the caller on the assumption one tenant is one vertical",
    "leadTimeDays": "no quote-level lead time exists (lead_time_days is procurement "
                    "lead time on item/vendor mappings). Proxied by quote validity: "
                    "expiration_date - quote date",
    "parentQuotationID": "the MRP schema has no revision chain — every quote is its "
                         "own family",
    "revisionNumber": "always 1, for the same reason",
    "negotiatedSalesPrice": "MaXXflow records one client-facing price; ingest falls "
                            "back to salesPrice",
}


def sql() -> str:
    """The executable statement, with the trailing prose stripped."""
    return SQL_PATH.read_text(encoding="utf-8").split(_FENCE)[0].strip()


def read_raw_export(tenant: str, industry: str, *, closed_only: bool = False,
                    conn=None) -> pd.DataFrame:
    """Raw export rows for one tenant, in the shape raw_ingest.build_line_frame eats.

    Isolation is ``SET search_path`` — handled by the data-access layer, never a
    ``WHERE tenant_id`` (maxxflow_data.engine rejects that outright).
    """
    from sqlalchemy import text

    stmt, params = text(sql()), {"industry": industry}
    if conn is not None:
        raw = pd.read_sql(stmt, conn, params=params)
    else:
        from maxxflow_data.engine import get_data_access
        with get_data_access().connect(tenant=tenant) as c:
            raw = pd.read_sql(stmt, c, params=params)

    if closed_only:
        raw = raw[raw["won"].notna()].copy()
    return raw


def quality_notes(raw: pd.DataFrame) -> list[dict]:
    """Per-column facts an operator needs BEFORE starting a training run.

    Deliberately not a pass/fail: a null listPrice is not invalid input, it just
    silently moves price_ratio onto a cost basis, and that is a decision someone
    should make knowingly rather than discover in the SHAP report.
    """
    n = len(raw)
    notes: list[dict] = []
    if not n:
        return [{"column": None, "level": "error",
                 "message": "the query returned no rows — is the tenant seeded?"}]

    def add(col, level, message):
        notes.append({"column": col, "level": level, "message": message})

    nulls = {c: int(raw[c].isna().sum()) for c in raw.columns}

    if nulls.get("listPrice", 0):
        k = nulls["listPrice"]
        add("listPrice", "warn" if k < n else "error",
            f"{k:,} of {n:,} rows ({k / n:.1%}) have no list price. Those lines are "
            f"scored against unit COST instead, which is a different scale from the "
            f"one the champion learned.")
    if nulls.get("leadTimeDays", 0):
        k = nulls["leadTimeDays"]
        add("leadTimeDays", "warn",
            f"{k:,} rows ({k / n:.1%}) have no expiration date, so the lead-time proxy "
            f"is null. Cleaning fills these with the training median.")
    for c in ("region", "materialSpec", "customerID", "salesRepID"):
        if nulls.get(c, 0):
            k = nulls[c]
            add(c, "warn", f"{k:,} rows ({k / n:.1%}) missing — falls back to the "
                           f"unknown level.")

    decided = int(raw["won"].notna().sum()) if "won" in raw.columns else 0
    if decided < n:
        add("won", "warn",
            f"{n - decided:,} of {n:,} lines belong to quotes that are still open. "
            f"Ingest's aging rule decides them; use closed_only=true to drop them "
            f"instead.")
    if decided:
        rate = float(raw.loc[raw["won"].notna(), "won"].mean())
        if rate in (0.0, 1.0):
            add("won", "error",
                f"every decided line has the same outcome (win rate {rate:.0%}) — "
                f"training needs both classes.")
        elif rate < 0.05 or rate > 0.95:
            add("won", "warn", f"win rate is {rate:.1%} — severely imbalanced.")

    if {"salesPrice", "listPrice"} <= set(raw.columns):
        ratio = (pd.to_numeric(raw["salesPrice"], errors="coerce")
                 / pd.to_numeric(raw["listPrice"], errors="coerce").replace(0, pd.NA))
        r = ratio.dropna()
        if len(r):
            add("salesPrice", "info",
                f"price ratio (salesPrice / listPrice) spans {r.quantile(0.01):.4f} to "
                f"{r.quantile(0.99):.4f}, median {r.median():.4f}. The served champion "
                f"is only informative between 0.7557 and 1.0882.")
    return notes
