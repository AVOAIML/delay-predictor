"""M1 CLI hooks (plan §11): fe / train / score / drift — local twins of the AML
steps. ML logic lives in features.py / train.py / score.py; this only wires IO."""

from __future__ import annotations

import os

import pandas as pd

from maxxflow_core.clock import get_clock
from maxxflow_core.errors import get_logger
from maxxflow_features.lake import get_lake
from maxxflow_mlops.drift import get_drift_reporter
from m1_quote.features import FEATURE_COLUMNS, build_features

log = get_logger("m1_quote.pipeline")
MODULE = "m1_quote"
_NUMERIC = [c for c in FEATURE_COLUMNS if c != "product_type_code"]


def fe(tenant: str = "demo") -> str:
    """bronze->silver->gold: read raw tables (DAL) -> features -> gold parquet (lake)."""
    from m1_quote.dal import read_quote_tables
    tables, md = read_quote_tables(tenant)
    feats = build_features(tables, md, get_clock())
    return get_lake().write_parquet(feats, "gold", "features", tenant=tenant, module=MODULE)


def train(tenant: str = "demo"):
    from m1_quote.train import train as _train
    return _train(tenant)


def score(tenant: str = "demo") -> int:
    from m1_quote.score import score as _score
    return _score(tenant)


def drift(tenant: str = "demo") -> str:
    """Compare the gold feature distribution (reference) vs a fresh build (current)."""
    lake = get_lake()
    try:
        reference = lake.read_parquet("gold", "features", tenant=tenant, module=MODULE)
    except FileNotFoundError:
        # No gold snapshot yet: rebuilding "reference" from the same live tables as
        # "current" would compare the data to itself and always report no drift,
        # even when that's wrong. Say so instead of running a meaningless report.
        # fsspec normalizes a missing object to FileNotFoundError on every backend
        # (local/S3/ADLS) this lake can point at — anything else is a real failure
        # and should surface, not be swallowed here.
        log.info("no gold feature snapshot for tenant=%r module=%r; skipping drift check", tenant, MODULE)
        return f"no baseline yet for tenant={tenant!r} — run `make fe MODULE={MODULE}` first"

    from m1_quote.dal import read_quote_tables
    tables, md = read_quote_tables(tenant)
    current = build_features(tables, md, get_clock())
    out_dir = os.environ.get("DRIFT_OUT", "reports")
    result = get_drift_reporter().report(reference[_NUMERIC], current[_NUMERIC],
                                         name=f"{tenant}_{MODULE}", out_dir=out_dir)
    return result["report_path"]
