"""Where M3's snapshot and outcome records live in the lake, and how a value
becomes strict JSON on the way there.

Layout, under the lake's own ``{lake_uri}/{tenant}/{module}/{layer}/`` prefix
(so tenant isolation is the first path segment, exactly as for M1/M2)::

    m3_production_delay/bronze/snapshots/dt=2026-09-24/WH_MO_00142-3f1c9a07be__20260924T021502123456Z.json
    m3_production_delay/bronze/outcomes/WH_MO_00142-3f1c9a07be.json
    m3_production_delay/bronze/_state/outcome_watermark.json

* **Snapshots are immutable, one per scoring event.** An MO is re-scored as it
  progresses, so the scoring instant is part of the name — a later score adds
  a file, it never replaces one. The ``dt=`` partition is the UTC date of that
  instant, so a training read globs a date range instead of listing every
  object in the tenant.
* **Outcomes are one per MO**, because an MO completes once. Rewriting one is
  harmless: the sweep that writes them re-derives the same record from the
  same rows.
* **The watermark sits under ``_state/``**, outside both record prefixes, so a
  glob over ``outcomes/*.json`` never reads it as a record.

MO references contain ``/`` (``WH/MO/00142``), which would read as a directory
separator. ``job_key`` makes a readable, filename-safe form and appends a short
hash of the exact reference, so two references that sanitize to the same text
can never share a file. The exact reference is always kept inside the record.

WHICH STORE — ``AZURE_STORAGE_CONTAINER``
-----------------------------------------
``local`` writes to the MinIO lake; any other value is the Azure container
(default ``dev``). The choice is made in ``Settings.container_lake_uri`` /
``container_lake_options`` — nothing in this package branches on it — and the
layout below the root is identical, so a path verified against MinIO is the
path Azure will get::

    local : s3://maxxflow-lake/demo/m3_production_delay/bronze/...
    dev   : abfss://dev@<account>.dfs.core.windows.net/maxxflow-lake/demo/m3_production_delay/bronze/...
"""

from __future__ import annotations

import datetime as _dt
import decimal
import hashlib
import math
import re
import uuid
from typing import Any

import numpy as np
import pandas as pd

MODULE = "m3_production_delay"
#: Medallion tier. These are raw, verbatim records — flattening into a
#: training table is a later (gold) step, once it is known which features matter.
LAYER = "bronze"

SNAPSHOTS_PREFIX = "snapshots"
OUTCOMES_PREFIX = "outcomes"
WATERMARK_NAME = "_state/outcome_watermark"

SNAPSHOT_SCHEMA = "m3.snapshot.v1"
OUTCOME_SCHEMA = "m3.outcome.v1"
WATERMARK_SCHEMA = "m3.outcome-watermark.v1"

#: How a non-finite float is written. Strict JSON has no infinity, and an
#: infinite ``material_shortfall_ratio`` (a component with zero stock) is a
#: real value the score depends on — so it is written as a string that
#: ``float()`` reads straight back. NaN means "no value" and is written as null.
POSITIVE_INFINITY = "inf"
NEGATIVE_INFINITY = "-inf"

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_KEY_HASH_CHARS = 10


def get_snapshot_lake(settings: Any | None = None):
    """The tenant artifact store selected by AZURE_STORAGE_CONTAINER.

    Raises ``ValueError`` when Azure is selected but not configured; callers
    that must not fail (the snapshot writer) catch it and log it.
    """
    from maxxflow_core.settings import get_settings
    from maxxflow_features.lake import LakeIO

    s = settings or get_settings()
    return LakeIO.at(s.container_lake_uri, s.container_lake_options)


def job_key(job_id: str) -> str:
    """Filename-safe, deterministic, collision-resistant form of an MO reference."""
    readable = _UNSAFE.sub("_", job_id).strip("_") or "job"
    digest = hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:_KEY_HASH_CHARS]
    return f"{readable}-{digest}"


def snapshot_name(job_id: str, scored_at: _dt.datetime) -> str:
    """Name for one scoring event of one MO. ``scored_at`` must be tz-aware."""
    if scored_at.tzinfo is None:
        raise ValueError("scored_at must be timezone-aware; a naive instant has no UTC date")
    utc = scored_at.astimezone(_dt.timezone.utc)
    return (
        f"{SNAPSHOTS_PREFIX}/dt={utc:%Y-%m-%d}/"
        f"{job_key(job_id)}__{utc:%Y%m%dT%H%M%S%fZ}"
    )


def outcome_name(job_id: str) -> str:
    return f"{OUTCOMES_PREFIX}/{job_key(job_id)}"


def to_jsonable(value: Any) -> Any:
    """Recursively convert ``value`` into something strict JSON accepts.

    Handles what actually arrives from the Risk Engine and pandas: NumPy
    scalars, ``uuid.UUID`` ids, ``Decimal`` quantities, pandas timestamps and
    ``NaT``, tuples, and non-string dict keys.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _float(float(value))
    if isinstance(value, decimal.Decimal):
        return _float(float(value))
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if value is pd.NaT:
        return None
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _float(value: float) -> float | str | None:
    if math.isnan(value):
        return None
    if math.isinf(value):
        return POSITIVE_INFINITY if value > 0 else NEGATIVE_INFINITY
    return value
