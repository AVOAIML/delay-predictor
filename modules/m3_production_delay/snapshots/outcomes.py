"""What actually happened to each MO — the label half of the training data.

A snapshot records the signals and the prediction at scoring time; on its own
it has features and no answer. This sweep supplies the answer: once an MO is
completed, it records when, against when it was planned to finish, with the
raw timings of every work order underneath.

HOW IT RUNS — a watermarked sweep
---------------------------------
Each run reads a bookmark (the newest ``completed_at`` it has already
processed), asks the tenant database for MOs completed after it, writes one
outcome record per MO, and only then moves the bookmark forward — to the
newest ``completed_at`` it saw, never to the wall clock. So:

* **Restartable.** A crash before the bookmark moves means the next run
  repeats the same MOs and rewrites the same records. Nothing is lost and
  nothing is duplicated (one record per MO, keyed by reference).
* **Replica-lag tolerant.** Each run looks back ``lookback`` before the
  bookmark, so an MO that reached the read replica late is still caught.
  Re-writing an outcome that already exists is harmless.
* **Read-only against the product.** The bookmark lives in the lake, not the
  database, because M3 reads a replica and owns no tables there.
* **Cancelled MOs are skipped.** A cancelled MO was neither late nor on time;
  counting it either way would corrupt the label.

THE LABEL — raw facts first, one swappable definition second
------------------------------------------------------------
``manufacturing_orders.scheduled_date`` is a single timestamp whose meaning
(planned start, or promised finish?) is not confirmed. So it is recorded but
NOT used. The label compares ``completed_at`` against the latest
``work_orders.scheduled_end`` — unambiguously a planned finish — and every raw
timestamp and duration is stored alongside, so a different definition can be
applied later over the same records without collecting them again.

Timestamps are written tz-aware in the platform's presentation timezone
(``Clock.localize``, the repo's single TZ-explicit rule). The bookmark keeps the
database's own naive value, so the next query compares like with like.
"""

from __future__ import annotations

import datetime as _dt
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from maxxflow_core.clock import get_clock
from maxxflow_core.errors import get_logger
from maxxflow_data.masterdata_map import load_md_map

from m3_production_delay.snapshots.store import (
    LAYER,
    MODULE,
    OUTCOME_SCHEMA,
    WATERMARK_NAME,
    WATERMARK_SCHEMA,
    get_snapshot_lake,
    outcome_name,
    to_jsonable,
)

log = get_logger("m3_production_delay.snapshots.outcomes")

DEFAULT_LOOKBACK = _dt.timedelta(hours=24)

#: Versioned so a training read can tell which rule produced a record's label.
LABEL_VERSION = "planned-finish-v1"
LABEL_DEFINITION = (
    "planned_finish = latest work_orders.scheduled_end of the MO; "
    "was_late = completed_at > planned_finish; hours_late = completed_at - planned_finish "
    "(negative = early). manufacturing_orders.scheduled_date is recorded but not used: "
    "whether it is a planned start or a promised finish is unconfirmed."
)

_CANCELLED = "CANCELLED"

# `(CAST(:since AS timestamp) IS NULL OR ...)` lets one statement serve the
# first run (no bookmark yet, sweep all history) and every run after it.
_COMPLETED_MOS = """
SELECT id, reference, scheduled_date, confirmed_at, completed_at, status_id
FROM manufacturing_orders
WHERE deleted_at IS NULL
  AND cancelled_at IS NULL
  AND completed_at IS NOT NULL
  AND (CAST(:since AS timestamp) IS NULL OR completed_at > CAST(:since AS timestamp))
ORDER BY completed_at
"""

# No operator columns: nothing here needs to know who did the work.
_WORK_ORDERS_FOR_MOS = """
SELECT mo_id, id, operation_id, expected_duration, real_duration,
       scheduled_start, scheduled_end, actual_start, actual_end, status_id
FROM work_orders
WHERE mo_id = ANY(CAST(:mo_ids AS uuid[]))
"""


@dataclass(frozen=True)
class SweepReport:
    tenant: str
    since: str | None
    found: int
    written: int
    skipped: int
    failed: int
    watermark_before: str | None
    watermark_after: str | None
    dry_run: bool


def label_outcome(completed_at: _dt.datetime | None, work_orders: list[dict]) -> dict:
    """The label for one completed MO, from database-native (naive) timestamps.

    Pure: no clock, no IO. ``work_orders`` are the MO's rows with raw values.
    """
    ends = [wo["scheduled_end"] for wo in work_orders if wo.get("scheduled_end") is not None]
    planned_finish = max(ends) if ends else None

    was_late: bool | None = None
    hours_late: float | None = None
    if completed_at is not None and planned_finish is not None:
        hours_late = (completed_at - planned_finish).total_seconds() / 3600.0
        was_late = hours_late > 0

    expected = [wo.get("expected_duration") for wo in work_orders]
    actual = [wo.get("real_duration") for wo in work_orders]
    expected_total = sum(v for v in expected if v is not None)
    actual_complete = bool(work_orders) and all(v is not None for v in actual)
    actual_total = sum(v for v in actual if v is not None)
    duration_overrun_ratio = (
        actual_total / expected_total if actual_complete and expected_total else None
    )

    return {
        "version": LABEL_VERSION,
        "definition": LABEL_DEFINITION,
        "planned_finish": planned_finish,
        "was_late": was_late,
        "hours_late": hours_late,
        "expected_minutes_total": expected_total,
        "actual_minutes_total": actual_total if actual_complete else None,
        "duration_overrun_ratio": duration_overrun_ratio,
    }


def build_outcome(
    mo: Mapping[str, Any],
    work_orders: list[dict],
    *,
    tenant: str,
    status_code: str | None,
    id_to_code: Mapping[str, str],
    swept_at: _dt.datetime,
) -> dict:
    """The outcome record for one completed MO, strict-JSON safe."""
    clock = get_clock()
    label = label_outcome(mo.get("completed_at"), work_orders)
    label["planned_finish"] = _aware(clock, label["planned_finish"])
    return to_jsonable(
        {
            "schema": OUTCOME_SCHEMA,
            "tenant": tenant,
            "job_id": mo["reference"],
            "status": status_code,
            "scheduled_date": _aware(clock, mo.get("scheduled_date")),
            "confirmed_at": _aware(clock, mo.get("confirmed_at")),
            "completed_at": _aware(clock, mo.get("completed_at")),
            "label": label,
            "work_orders": [_work_order(wo, clock, id_to_code) for wo in work_orders],
            "swept_at": swept_at.isoformat(),
        }
    )


def sweep_outcomes(
    tenant: str,
    *,
    lookback: _dt.timedelta = DEFAULT_LOOKBACK,
    dry_run: bool = False,
    lake: Any | None = None,
    data_access: Any | None = None,
) -> SweepReport:
    """Write an outcome for every MO completed since the bookmark, then move it."""
    if lake is None:
        lake = get_snapshot_lake()
    if data_access is None:
        from maxxflow_data.engine import get_data_access

        data_access = get_data_access()

    clock = get_clock()
    mark = lake.read_json(LAYER, WATERMARK_NAME, tenant=tenant, module=MODULE)
    watermark = _parse_db_timestamp(mark.get("completed_at")) if mark else None
    since = watermark - lookback if watermark is not None else None

    mos = _records(data_access.query(_COMPLETED_MOS, {"since": since}, tenant=tenant))
    watermark_before = watermark.isoformat() if watermark else None
    if not mos:
        log.info(
            "m3_outcomes nothing completed tenant=%s since=%s", tenant, since
        )
        return SweepReport(
            tenant=tenant,
            since=since.isoformat() if since else None,
            found=0,
            written=0,
            skipped=0,
            failed=0,
            watermark_before=watermark_before,
            watermark_after=watermark_before,
            dry_run=dry_run,
        )

    md = load_md_map(data_access, tenant)
    id_to_code = {str(v): k[1] for k, v in md.code_to_id.items()}
    work_orders_by_mo: dict[str, list[dict]] = defaultdict(list)
    for wo in _records(
        data_access.query(
            _WORK_ORDERS_FOR_MOS, {"mo_ids": [str(mo["id"]) for mo in mos]}, tenant=tenant
        )
    ):
        work_orders_by_mo[str(wo["mo_id"])].append(wo)

    swept_at = clock.now_utc()
    written = skipped = failed = 0
    newest = watermark
    for mo in mos:
        status_code = id_to_code.get(str(mo.get("status_id")))
        if status_code == _CANCELLED:
            skipped += 1
            continue
        record = build_outcome(
            mo,
            work_orders_by_mo.get(str(mo["id"]), []),
            tenant=tenant,
            status_code=status_code,
            id_to_code=id_to_code,
            swept_at=swept_at,
        )
        completed = mo["completed_at"]
        newest = completed if newest is None or completed > newest else newest
        if dry_run:
            log.info(
                "m3_outcomes DRY RUN job_id=%s was_late=%s hours_late=%s",
                record["job_id"],
                record["label"]["was_late"],
                record["label"]["hours_late"],
            )
            continue
        try:
            lake.write_json(
                record, LAYER, outcome_name(mo["reference"]), tenant=tenant, module=MODULE
            )
            written += 1
        except Exception:
            failed += 1
            log.exception(
                "m3_outcomes write failed job_id=%s tenant=%s", mo["reference"], tenant
            )

    # The bookmark moves only when every record in this window landed. One
    # failure holds it where it was, so the next run retries the whole window.
    watermark_after = watermark_before
    if not dry_run and failed == 0 and newest is not None and newest != watermark:
        lake.write_json(
            {
                "schema": WATERMARK_SCHEMA,
                "completed_at": newest.isoformat(),
                "note": "database-native timestamp (no timezone), compared as-is",
                "updated_at": swept_at.isoformat(),
            },
            LAYER,
            WATERMARK_NAME,
            tenant=tenant,
            module=MODULE,
        )
        watermark_after = newest.isoformat()

    log.info(
        "m3_outcomes tenant=%s store=%s since=%s found=%d written=%d skipped=%d failed=%d "
        "watermark_before=%s watermark_after=%s dry_run=%s",
        tenant,
        getattr(lake, "root", "?"),
        since,
        len(mos),
        written,
        skipped,
        failed,
        watermark_before,
        watermark_after,
        dry_run,
    )
    return SweepReport(
        tenant=tenant,
        since=since.isoformat() if since else None,
        found=len(mos),
        written=written,
        skipped=skipped,
        failed=failed,
        watermark_before=watermark_before,
        watermark_after=watermark_after,
        dry_run=dry_run,
    )


def _work_order(wo: Mapping[str, Any], clock, id_to_code: Mapping[str, str]) -> dict:
    expected = wo.get("expected_duration")
    actual = wo.get("real_duration")
    scheduled_end = wo.get("scheduled_end")
    actual_end = wo.get("actual_end")
    return {
        "work_order_id": wo.get("id"),
        "operation_id": wo.get("operation_id"),
        "status": id_to_code.get(str(wo.get("status_id"))),
        "expected_duration_minutes": expected,
        "actual_duration_minutes": actual,
        "scheduled_start": _aware(clock, wo.get("scheduled_start")),
        "scheduled_end": _aware(clock, scheduled_end),
        "actual_start": _aware(clock, wo.get("actual_start")),
        "actual_end": _aware(clock, actual_end),
        "finished_late": (
            actual_end > scheduled_end
            if actual_end is not None and scheduled_end is not None
            else None
        ),
        "duration_overrun_ratio": actual / expected if actual is not None and expected else None,
    }


def _records(frame: pd.DataFrame) -> list[dict]:
    """Rows as dicts with pandas' missing markers turned into None and
    timestamps into plain datetimes, so the logic above compares real values."""
    rows = []
    for row in frame.to_dict("records"):
        clean = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                value = value.to_pydatetime()
            elif value is pd.NaT or (isinstance(value, float) and pd.isna(value)):
                value = None
            clean[key] = value
        rows.append(clean)
    return rows


def _aware(clock, value: _dt.datetime | None) -> _dt.datetime | None:
    if value is None:
        return None
    if value.tzinfo is not None:
        return value
    return clock.localize(value)


def _parse_db_timestamp(value: str | None) -> _dt.datetime | None:
    return _dt.datetime.fromisoformat(value) if value else None
