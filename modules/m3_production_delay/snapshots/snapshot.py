"""One immutable record per scoring event: what M3 saw, and what it said.

Written after the advisory writeback, once per reviewed job, so a later
training step can pair *the signals at scoring time* with *what actually
happened* (``outcomes.py``). Every signal value is kept — including the ones
that did not fire, which never appear in the published why-lines but are
exactly what a weight fit needs.

**What is deliberately left out.** The Risk Engine's operation dict carries
``operators``: pseudonymised operator tokens, a ``name`` field, and each
operator's last ten work orders. None of that goes into the lake. Operator
identity must never reach bronze; the only operator facts kept are the ones
the score already reduced them to — ``operator_pace_ratio`` and
``operator_count``. Fields are therefore copied from an allow-list, never by
dumping the dict, so a field the engine adds later cannot leak in unreviewed.
Vendor purchase-order histories are likewise reduced to the
``vendor_lead_time_ratio`` the score actually used.

**Failure never reaches the caller.** The snapshot is collected for future
training; the planner-facing writeback has already happened by the time this
runs. A lake that is down costs a logged error per job, not a failed review.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from maxxflow_core.clock import get_clock
from maxxflow_core.errors import get_logger

from m3_production_delay.review.evidence import supplier_reliability
from m3_production_delay.review.schemas import ValidatedInsight
from m3_production_delay.snapshots.store import (
    LAYER,
    MODULE,
    SNAPSHOT_SCHEMA,
    snapshot_name,
    to_jsonable,
)

log = get_logger("m3_production_delay.snapshots.snapshot")

#: Per-operation fields copied from the scored job. Everything the composite
#: score and the overrun projection read, plus the identifiers to join on.
OPERATION_FIELDS: tuple[str, ...] = (
    "operation_id",
    "operation_type",
    "status",
    "depends_on_operation_ids",
    "expected_duration_minutes",
    "actual_duration_minutes",
    "job_quantity",
    "current_done_quantity",
    "operator_count",
    "time_overrun_ratio",
    "operator_pace_ratio",
    "material_shortfall_ratio",
    "predecessor_time_overrun_ratio",
    "predicted_overrun_hours",
    "composite_risk_score",
    "is_delayed",
)
COMPONENT_FIELDS: tuple[str, ...] = (
    "component_id",
    "name",
    "required_quantity",
    "available_quantity",
)
VENDOR_FIELDS: tuple[str, ...] = ("vendor_id", "name", "vendor_lead_time_ratio")


@dataclass(frozen=True)
class SnapshotReport:
    written: int
    failed: int


def build_snapshot(
    job: Mapping[str, Any] | None,
    insight: ValidatedInsight,
    *,
    tenant: str,
    risk_weights: Mapping[str, float],
    threshold: float,
    scored_at: _dt.datetime,
    written_at: _dt.datetime,
) -> dict:
    """The record for one scoring event, already strict-JSON safe."""
    operations = (job or {}).get("operations") or []
    return to_jsonable(
        {
            "schema": SNAPSHOT_SCHEMA,
            "tenant": tenant,
            "job_id": insight.job_id,
            "scored_at": scored_at.isoformat(),
            "written_at": written_at.isoformat(),
            "model_version": insight.model_version,
            "delay_threshold": threshold,
            "risk_weights": dict(risk_weights),
            "operations": [_operation(op) for op in operations],
            "insight": insight.to_dict(),
        }
    )


def write_snapshots(
    scored_jobs: Sequence[Mapping[str, Any]],
    insights: Sequence[ValidatedInsight],
    *,
    tenant: str,
    risk_weights: Mapping[str, float],
    threshold: float,
    lake: Any | None = None,
) -> SnapshotReport:
    """Write one snapshot per insight. Never raises."""
    if not insights:
        return SnapshotReport(written=0, failed=0)

    clock = get_clock()
    jobs_by_id = {job.get("job_id"): job for job in scored_jobs}
    written = failed = 0
    try:
        if lake is None:
            from maxxflow_features.lake import get_lake

            lake = get_lake()
    except Exception:
        log.exception("m3_snapshots lake unavailable tenant=%s jobs=%d", tenant, len(insights))
        return SnapshotReport(written=0, failed=len(insights))

    for insight in insights:
        try:
            job = jobs_by_id.get(insight.job_id)
            if job is None:
                log.warning(
                    "m3_snapshots no scored job for insight job_id=%s tenant=%s; "
                    "writing the insight without operation signals",
                    insight.job_id,
                    tenant,
                )
            scored_at = _scored_at(insight, clock)
            record = build_snapshot(
                job,
                insight,
                tenant=tenant,
                risk_weights=risk_weights,
                threshold=threshold,
                scored_at=scored_at,
                written_at=clock.now_utc(),
            )
            lake.write_json(
                record,
                LAYER,
                snapshot_name(insight.job_id, scored_at),
                tenant=tenant,
                module=MODULE,
            )
            written += 1
        except Exception:
            failed += 1
            log.exception(
                "m3_snapshots write failed job_id=%s tenant=%s", insight.job_id, tenant
            )

    log.info("m3_snapshots written=%d failed=%d tenant=%s", written, failed, tenant)
    return SnapshotReport(written=written, failed=failed)


def _scored_at(insight: ValidatedInsight, clock) -> _dt.datetime:
    """The review's own clock stamp, so the snapshot names the instant the
    insight was produced — not whenever the write happened to run."""
    if insight.generated_at:
        parsed = _dt.datetime.fromisoformat(insight.generated_at)
        if parsed.tzinfo is not None:
            return parsed
    return clock.as_of()


def _operation(op: Mapping[str, Any]) -> dict:
    components = op.get("components") or []
    out = {field: op.get(field) for field in OPERATION_FIELDS}
    # The engine computes this per operation but does not attach it to the
    # operation dict; recompute it from the same component ratios, with the
    # same definition Section 3's evidence uses.
    out["supplier_reliability"] = supplier_reliability(components)
    out["components"] = [_component(c) for c in components]
    return out


def _component(component: Mapping[str, Any]) -> dict:
    out = {field: component.get(field) for field in COMPONENT_FIELDS}
    vendor = component.get("vendor")
    out["vendor"] = (
        {field: vendor.get(field) for field in VENDOR_FIELDS} if vendor else None
    )
    return out
