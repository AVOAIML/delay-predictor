"""Local-only demo bridge: serves the real M3 delay-insights read over HTTP,
with no auth, for the frontend/ demo to call while iterating locally. Also
exposes a "create a job and score it now" endpoint so the frontend's Create
MO form can exercise the full flow — real rows in, real pipeline run, real
insight back — without a second terminal command per test. A third endpoint
runs the real Weight Agent (admin-configured or cold-start) and re-scores a
job with exactly the weights it resolved, for the "Admin: Configure Weights"
panel.

services/configurator/app.py's actual `GET /api/{tenant}/delay-insights`
route is real and reads the same table/column this does — but it sits behind
`install_security`'s JWT + DB-backed RBAC (services/configurator/security.py),
which expects a shared-product `public` schema (public.master_data,
public.user_tenants, public.roles, public.permissions, users.is_ci_admin,
users.upn) that this AI/ML repo's own `db-provision` never creates — it only
defines `public.users`/`public.tenants`
(libs/maxxflow_data/schema_def.py PUBLIC_TABLES). That schema belongs to the
main MaXXFlow product's own database, not this repo, so it doesn't exist in
a local-only checkout. Bearer-token auth is therefore not exercisable here
without hand-building unrelated product infrastructure.

This script exists to still let the frontend talk to something real: same
read SQL as `services/configurator/app.py`'s `delay_insights()`, same DAL
(`maxxflow_data.engine`); the create+score route below drives the actual
`m3_production_delay.review.pipeline.run()` — just without an auth
middleware in front of either. Do not point this at anything but a local dev
Postgres.

Usage:
    uv run uvicorn scripts.m3_demo_api:app --port 8010 --reload
"""

from __future__ import annotations

import random

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:  # `python scripts/m3_demo_api.py`-style invocation puts scripts/ on sys.path
    from _m3_demo_common import ensure_reference_data, insert_job
except ImportError:  # `uvicorn scripts.m3_demo_api:app`-style invocation does not
    from scripts._m3_demo_common import ensure_reference_data, insert_job

app = FastAPI(title="M3 demo bridge (unauthenticated, local only)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

M3_ADVISORY_KEY = "ai_delay_insight"


@app.get("/api/{tenant}/delay-insights")
def delay_insights(tenant: str, job: str | None = Query(default=None)):
    """Verbatim copy of services/configurator/app.py's `delay_insights` query,
    minus request.state.auth / the x-tenant-slug cross-check — there is no
    authenticated caller here to check it against."""
    from maxxflow_data.engine import get_data_access

    sql = (
        "SELECT reference, custom_elements -> :key AS insight "
        "FROM manufacturing_orders "
        "WHERE deleted_at IS NULL AND custom_elements ? :key"
    )
    params: dict = {"key": M3_ADVISORY_KEY}
    if job:
        sql += " AND reference = :reference"
        params["reference"] = job
    sql += " ORDER BY reference"

    try:
        frame = get_data_access().query(sql, params, tenant=tenant)
    except Exception as exc:
        raise HTTPException(409, f"cannot read delay insights: {exc}") from exc

    rows = [
        {"job_id": record["reference"], "insight": record["insight"]}
        for record in frame.to_dict(orient="records")
    ]
    if job and not rows:
        raise HTTPException(
            404,
            f"no cached delay insight for {job!r} — run "
            f"`uv run python -m m3_production_delay.review --tenant {tenant} --threshold <t> "
            f"--job {job!r}` first",
        )
    return {"tenant": tenant, "count": len(rows), "insights": rows}


class ComponentInput(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    required_quantity: float = Field(gt=0)
    available_quantity: float = Field(ge=0)


class CreateMoRequest(BaseModel):
    product: str = Field(min_length=1, max_length=255)
    quantity: float = Field(gt=0)
    operation_name: str = Field(min_length=1, max_length=255)
    expected_duration_hours: float = Field(gt=0, le=1000)
    actual_duration_hours: float = Field(ge=0, le=1000)
    threshold: float = Field(default=1.0, gt=0)
    components: list[ComponentInput] = Field(default_factory=list, max_length=20)


@app.post("/api/{tenant}/demo-manufacturing-orders")
def create_and_score(tenant: str, body: CreateMoRequest):
    """Inserts one new job into the tenant schema (real MRP tables, via
    `_m3_demo_common.insert_job`) and immediately runs the real M3 pipeline
    against it — the full flow a "Plan" click would trigger, minus the
    product's own MO-creation UI, which lives outside this repo."""
    from maxxflow_data.engine import get_data_access
    from m3_production_delay.review.pipeline import run as run_delay_review

    job_reference = f"WH/MO/DEMO-{random.randint(10000, 99999)}"
    da = get_data_access()

    try:
        with da.transaction(tenant=tenant) as conn:
            ensure_reference_data(conn)
            insert_job(
                conn,
                job_reference=job_reference,
                quantity=body.quantity,
                operation_name=body.operation_name,
                expected_duration_minutes=body.expected_duration_hours * 60,
                actual_duration_minutes=body.actual_duration_hours * 60,
                components=[c.model_dump() for c in body.components],
            )
    except Exception as exc:
        raise HTTPException(409, f"could not insert job: {exc}") from exc

    try:
        insights = run_delay_review(tenant=tenant, threshold=body.threshold, job_references=[job_reference])
    except Exception as exc:
        raise HTTPException(409, f"job was created but scoring failed: {exc}") from exc

    if not insights:
        raise HTTPException(409, "job was created but produced no scorable insight")

    return {"tenant": tenant, "job_id": job_reference, "insight": insights[0].to_dict()}


class ResolveWeightsRequest(BaseModel):
    # None ("Skip" in the UI) takes the Weight Agent's cold-start path: no
    # tenant_description is supplied either, so — with the local stub LLM
    # returning text, not JSON — it always bottoms out at source="prior"
    # (see orchestrator.py's module docstring). A dict ("Add" in the UI)
    # takes the configured path and must carry exactly the five SIGNAL_ORDER
    # keys summing to 10000, each within its own Bounds (config.py
    # DEFAULT_BOUNDS_BP) — WeightAgent.resolve() validates this itself.
    configured_bp: dict[str, int] | None = None
    job_reference: str = Field(min_length=1)
    threshold: float = Field(default=1.0, gt=0)


@app.post("/api/{tenant}/demo-weights")
def resolve_weights_and_rescore(tenant: str, body: ResolveWeightsRequest):
    """Runs the REAL Weight Agent (`ProductionDelayOrchestrator.resolve_weights`)
    with either admin-supplied `configured_bp` or none at all (cold start),
    then re-scores one already-seeded job with exactly the weights that came
    back — the same three rule-engine calls `review.pipeline.run()` makes
    internally, just with the resolution done here instead of re-derived, so
    the admin can see the Weight Agent's actual output before it disappears
    into a score."""
    from m3_production_delay.llm_agents.weight_agent.exceptions import AllSignalsUnavailableError
    from m3_production_delay.llm_agents.weight_agent.models import SIGNAL_ORDER
    from m3_production_delay.orchestrator import ProductionDelayOrchestrator, WeightAgentRequest
    from m3_production_delay.review.publish import publish_insights
    from m3_production_delay.rule_engine.dal import read_delay_tables
    from m3_production_delay.rule_engine.elements import (
        calculate_delay_elements_for_jobs,
        weights_bp_to_risk_weights,
    )
    from m3_production_delay.rule_engine.rollup import build_job_rollups

    orchestrator = ProductionDelayOrchestrator()
    try:
        resolution = orchestrator.resolve_weights(
            WeightAgentRequest(
                tenant_id=tenant,
                availability={signal: True for signal in SIGNAL_ORDER},
                configured_bp=body.configured_bp,
            )
        )
    except AllSignalsUnavailableError as exc:
        # The only exception WeightAgent.resolve() ever actually raises — an
        # invalid `configured_bp` (bad sum, out-of-bounds signal) is instead
        # caught INSIDE resolve() and turned into a `fallback_reasons` entry
        # on a source="prior" result (see resolver.py's `configured_weights_
        # invalid` catch), so there is no WeightConfigError to catch here.
        raise HTTPException(422, f"weight agent: {exc}") from exc

    risk_weights = weights_bp_to_risk_weights(resolution.weights_bp)

    tables, md = read_delay_tables(tenant)
    rollups = build_job_rollups(tables, md, job_references=[body.job_reference])
    if not rollups:
        raise HTTPException(404, f"no such job {body.job_reference!r} — create or seed it first")

    scored = calculate_delay_elements_for_jobs(rollups, risk_weights=risk_weights, delay_threshold=body.threshold)
    insights = orchestrator.review_jobs(scored, weights=risk_weights, threshold=body.threshold)
    publish_insights(insights, tenant=tenant, dry_run=False)

    return {
        "tenant": tenant,
        "weight_resolution": resolution.to_json_dict(),
        "risk_weights": risk_weights,
        "job_id": body.job_reference,
        "insight": insights[0].to_dict() if insights else None,
    }
