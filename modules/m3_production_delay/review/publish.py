"""Writeback for the validated insight, following M1/M2's advisory pattern
(``m1_quote/score.py``, ``m2_inventory/batch_scoring.py``): merge a namespaced
JSON blob into the entity's ``customElements`` with ``COALESCE(...) || jsonb``,
and record the outputs a human should know about in ``audit_logs``.

Target: **``manufacturing_orders.customElements`` under ``ai_delay_insight``**.
``work_orders`` has no ``customElements`` column, so the MO row is where a
per-job advisory can live without a migration — the same reasoning, and the
same table, the module's per-operation ``ai_delay`` advisory uses. This writer
touches only its own ``ai_delay_insight`` key; the ``||`` merge leaves every
other key on the row, including ``ai_delay``, exactly as it was.

Audited statuses are the three where the panel is not showing a fully
validated explanation — ``fallback_template`` (the judge could not be reached,
read, or convinced), ``rejected`` (a deterministic validator failed) and
``suppressed_not_scorable`` (no operation had enough logged progress). Those
are exactly the cases M1/M2 audit rather than drop silently.
"""

from __future__ import annotations

import json

import sqlalchemy as sa

from maxxflow_core.clock import get_clock
from maxxflow_core.errors import get_logger
from maxxflow_core.jsonutil import json_default

from m3_production_delay.review.schemas import ValidatedInsight

log = get_logger("m3_production_delay.review.publish")

MODULE = "m3_production_delay"
#: Human-readable label for ``audit_logs.module``, like M1's "Smart Quote
#: Optimizer" and M2's "Predictive Inventory Alerts".
MODULE_DISPLAY_NAME = "Production Delay Insights"
#: Version of the review contract itself. Not an MLflow model version — there
#: is no registered model behind Section 3; the judge is an API call and the
#: explanation is deterministic.
MODEL_VERSION = "m3-review-v1"
#: The single key this writer owns inside ``customElements``.
ADVISORY_KEY = "ai_delay_insight"

_UPDATE_SQL = sa.text(
    "UPDATE manufacturing_orders "
    "SET custom_elements = COALESCE(custom_elements,'{}'::jsonb) || CAST(:payload AS jsonb) "
    "WHERE reference = :reference AND deleted_at IS NULL"
)

# entity_id is a uuid column, and a review is keyed by the MO's business
# reference, so the id is resolved in the INSERT itself rather than carried
# through this module as a second identifier.
_AUDIT_SQL = sa.text(
    "INSERT INTO audit_logs (module, action, entity_type, entity_id, metadata, timestamp) "
    "SELECT :module, 'review', 'ManufacturingOrder', mo.id, CAST(:metadata AS jsonb), now() "
    "FROM manufacturing_orders mo WHERE mo.reference = :reference AND mo.deleted_at IS NULL"
)


def advisory_payload(insight: ValidatedInsight) -> dict:
    """The exact JSON object merged into ``customElements``.

    ``scored_at`` comes from the platform clock (TZ-explicit, override-driven
    in tests) rather than ``now()``, so a replayed run writes the same
    timestamp it would have written live.
    """
    values = insight.to_dict()
    values["scored_at"] = get_clock().as_of().isoformat()
    return {ADVISORY_KEY: values}


def _audit_metadata(insight: ValidatedInsight) -> dict:
    """Why this insight was audited, in the terms a reader can act on — no
    line text, since a rejected or suppressed insight has none worth quoting."""
    return {
        "job_reference": insight.job_id,
        "status": insight.status,
        "risk_score": insight.risk_score,
        "is_delayed": insight.is_delayed,
        "delay_threshold": insight.delay_threshold,
        "attempts": insight.attempts,
        "judge_parse_error": insight.judge.parse_error,
        "issues": [issue.to_dict() for issue in insight.issues if issue.severity != "info"],
        "model_version": insight.model_version,
    }


def _dumps(payload: dict) -> str:
    """``allow_nan=False`` is the last line of defence: the schemas coerce
    every infinity at construction, and this makes a leak fail loudly here
    instead of writing an ``Infinity`` literal no JSON reader can parse."""
    return json.dumps(payload, default=json_default, allow_nan=False)


def publish_insights(
    insights: list[ValidatedInsight], *, tenant: str = "demo", dry_run: bool = False
) -> int:
    """Write every insight to its manufacturing order. Returns the number of
    rows updated (in a dry run, the number that would have been).

    One transaction for the batch, tenant-scoped through the DAL's
    ``search_path`` isolation — no query here names a tenant column.
    """
    if not insights:
        return 0
    if dry_run:
        for insight in insights:
            log.info(
                "m3_review DRY RUN job_id=%s status=%s lines=%d",
                insight.job_id,
                insight.status,
                len(insight.why_lines),
            )
        return len(insights)

    from maxxflow_data.engine import get_data_access

    data_access = get_data_access()
    written = 0
    audited = 0
    with data_access.transaction(tenant=tenant) as conn:
        for insight in insights:
            result = conn.execute(
                _UPDATE_SQL,
                {"payload": _dumps(advisory_payload(insight)), "reference": insight.job_id},
            )
            if result.rowcount:
                written += result.rowcount
            else:
                # The job was rolled up from this schema, so a miss means the
                # MO was deleted between the read and the write — worth a log,
                # not worth failing the batch for.
                log.warning(
                    "m3_review writeback matched no manufacturing order job_id=%s tenant=%s",
                    insight.job_id,
                    tenant,
                )
            if insight.needs_audit:
                conn.execute(
                    _AUDIT_SQL,
                    {
                        "module": MODULE_DISPLAY_NAME,
                        "metadata": _dumps(_audit_metadata(insight)),
                        "reference": insight.job_id,
                    },
                )
                audited += 1

    log.info(
        "m3_review published insights=%d rows=%d audited=%d tenant=%s",
        len(insights),
        written,
        audited,
        tenant,
    )
    return written
