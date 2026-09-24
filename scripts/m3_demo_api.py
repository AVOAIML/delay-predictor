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

import json
import random
from typing import Any

import sqlalchemy as sa
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:  # `python scripts/m3_demo_api.py`-style invocation puts scripts/ on sys.path
    from _m3_demo_common import ensure_reference_data, insert_configured_job
except ImportError:  # `uvicorn scripts.m3_demo_api:app`-style invocation does not
    from scripts._m3_demo_common import ensure_reference_data, insert_configured_job

app = FastAPI(title="M3 demo bridge (unauthenticated, local only)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

M3_ADVISORY_KEY = "ai_delay_insight"


def _json_object(value: Any) -> dict:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


def _date_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%d/%m/%Y")
    return str(value)


def _timestamp_text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _stage(record: dict) -> str:
    status = str(record.get("status_code") or record.get("status_name") or "").upper()
    if not _is_missing(record.get("completed_at")) or status in {"DONE", "COMPLETED", "CLOSED"}:
        return "done"
    if not _is_missing(record.get("confirmed_at")) or status not in {"", "DRAFT"}:
        return "confirmed"
    return "draft"


def _load_manufacturing_orders(tenant: str, references: list[str] | None = None) -> list[dict]:
    """Return the MO page's complete read model from tenant-scoped DB rows.

    Only active orders are shown. Completed MOs are operator-history inputs,
    not current work for the planner dropdown. No presentation field is
    reconstructed in React after creation; the POST route calls this same
    reader and returns the newly persisted order.
    """
    from maxxflow_data.engine import get_data_access

    da = get_data_access()
    where = "WHERE mo.deleted_at IS NULL AND mo.completed_at IS NULL"
    params: dict[str, Any] = {"key": M3_ADVISORY_KEY}
    if references:
        where += " AND mo.reference = ANY(:references)"
        params["references"] = references

    orders_frame = da.query(
        f"""
        SELECT mo.id, mo.reference, mo.product_id, mo.bom_id, mo.quantity,
               mo.scheduled_date, mo.confirmed_at, mo.completed_at, mo.created_at,
               mo.custom_elements, p.name AS product_name, b.name AS bom_name,
               status.name AS status_name, status.code AS status_code
        FROM manufacturing_orders mo
        LEFT JOIN products p ON p.id = mo.product_id AND p.deleted_at IS NULL
        LEFT JOIN boms b ON b.id = mo.bom_id AND b.deleted_at IS NULL
        LEFT JOIN master_data status ON status.id = mo.status_id
        {where}
        ORDER BY mo.created_at DESC, mo.reference
        """,
        params,
        tenant=tenant,
    )
    if orders_frame.empty:
        return []

    components_frame = da.query(
        f"""
        SELECT mc.mo_id,
               COALESCE(i.item_name, component_product.name, 'Component') AS component_name,
               mc.required_qty,
               COALESCE(i.available_quantity, component_product.on_hand, mc.reserved_qty, 0)
                   AS available_qty,
               availability.name AS availability_name,
               availability.code AS availability_code
        FROM mo_components mc
        JOIN manufacturing_orders mo ON mo.id = mc.mo_id
        LEFT JOIN items i ON i.id = mc.item_id AND i.deleted_at IS NULL
        LEFT JOIN products component_product
               ON component_product.id = mc.product_id AND component_product.deleted_at IS NULL
        LEFT JOIN master_data availability ON availability.id = mc.availability_id
        {where}
        ORDER BY mc.created_at, mc.id
        """,
        params,
        tenant=tenant,
    )
    work_orders_frame = da.query(
        f"""
        SELECT wo.mo_id, wo.id, op."operationName" AS operation_name,
               wc.name AS work_center_name,
               status.name AS status_name, status.code AS status_code,
               wo.quantity, wo.units_done, wo.expected_duration, wo.real_duration,
               wo.scheduled_start, wo.scheduled_end, wo.actual_start, wo.actual_end
        FROM work_orders wo
        JOIN manufacturing_orders mo ON mo.id = wo.mo_id
        LEFT JOIN operations op ON op.id = wo.operation_id
        LEFT JOIN work_centers wc ON wc.id = wo.work_center_id
        LEFT JOIN master_data status ON status.id = wo.status_id
        {where}
        ORDER BY wo.scheduled_start NULLS LAST, wo.created_at, wo.id
        """,
        params,
        tenant=tenant,
    )

    components_by_mo: dict[str, list[dict]] = {}
    for component in components_frame.to_dict(orient="records"):
        required = float(component["required_qty"])
        available = float(component["available_qty"])
        availability = "Short" if available < required else "Available"
        if available <= 0 and required > 0:
            availability = "Not Available"
        components_by_mo.setdefault(str(component["mo_id"]), []).append({
            "product": component["component_name"],
            "availability": availability,
            "toConsume": required,
            "availableQuantity": available,
        })

    work_orders_by_mo: dict[str, list[dict]] = {}
    for work_order in work_orders_frame.to_dict(orient="records"):
        work_orders_by_mo.setdefault(str(work_order["mo_id"]), []).append({
            "id": str(work_order["id"]),
            "operationName": work_order.get("operation_name") or "Operation",
            "workCenter": work_order.get("work_center_name") or "Work Center",
            "status": work_order.get("status_name") or work_order.get("status_code") or "Unknown",
            "quantity": float(work_order["quantity"]),
            "unitsDone": float(work_order["units_done"]),
            "expectedDurationMinutes": int(work_order["expected_duration"]),
            "actualDurationMinutes": (
                int(work_order["real_duration"])
                if not _is_missing(work_order.get("real_duration")) else None
            ),
            "scheduledStart": _timestamp_text(work_order.get("scheduled_start")),
            "scheduledEnd": _timestamp_text(work_order.get("scheduled_end")),
            "actualStart": _timestamp_text(work_order.get("actual_start")),
            "actualEnd": _timestamp_text(work_order.get("actual_end")),
        })

    orders: list[dict] = []
    for record in orders_frame.to_dict(orient="records"):
        custom = _json_object(record.get("custom_elements"))
        mo_id = str(record["id"])
        created_at = record.get("created_at")
        activity = [{
            "actor": "System",
            "initials": "SY",
            "timestamp": _timestamp_text(created_at) or "",
            "title": "Manufacturing Order Created",
        }]
        if not _is_missing(record.get("confirmed_at")):
            activity.insert(0, {
                "actor": "System",
                "initials": "SY",
                "timestamp": _timestamp_text(record["confirmed_at"]) or "",
                "title": "Manufacturing Order Confirmed",
            })
        orders.append({
            "reference": record["reference"],
            "product": record.get("product_name") or custom.get("demo_product") or "Unresolved product",
            "quantity": float(record["quantity"]),
            "bom": record.get("bom_name") or custom.get("demo_bom") or "No BOM",
            "scheduledDate": _date_text(record.get("scheduled_date")),
            "stage": _stage(record),
            "components": components_by_mo.get(mo_id, []),
            "workOrders": work_orders_by_mo.get(mo_id, []),
            "activity": activity,
            "insight": custom.get(M3_ADVISORY_KEY),
        })
    return orders


@app.get("/api/{tenant}/demo-manufacturing-orders")
def list_manufacturing_orders(tenant: str):
    """List active MOs with their DB-backed page data and cached insight."""
    try:
        orders = _load_manufacturing_orders(tenant)
    except Exception as exc:
        raise HTTPException(409, f"cannot read manufacturing orders: {exc}") from exc
    return {"tenant": tenant, "count": len(orders), "orders": orders}


@app.get("/api/{tenant}/demo-manufacturing-order-options")
def manufacturing_order_options(tenant: str):
    """Product/BOM defaults used by the DB-driven local creation form."""
    from maxxflow_data.engine import get_data_access

    da = get_data_access()
    try:
        products = da.query("""
            SELECT p.id, p.sku, p.name
            FROM products p WHERE p.deleted_at IS NULL
            ORDER BY p.name
        """, tenant=tenant).to_dict(orient="records")
        boms = da.query("""
            SELECT b.id, b.product_id, b.code, b.name
            FROM boms b WHERE b.deleted_at IS NULL
            ORDER BY b.name
        """, tenant=tenant).to_dict(orient="records")
        components = da.query("""
            SELECT bc.bom_id, bc.item_id,
                   COALESCE(i.item_name, component_product.name, 'Component') AS name,
                   bc.quantity AS required_quantity,
                   COALESCE(i.available_quantity, component_product.on_hand, 0) AS available_quantity
            FROM bom_components bc
            LEFT JOIN items i ON i.id = bc.item_id AND i.deleted_at IS NULL
            LEFT JOIN products component_product
                   ON component_product.id = bc.product_id AND component_product.deleted_at IS NULL
            ORDER BY bc.created_at, bc.id
        """, tenant=tenant).to_dict(orient="records")
        operations = da.query("""
            SELECT op.id, op.bom_id, op."operationName" AS name,
                   wc.id AS work_center_id, wc.name AS work_center_name,
                   op."estimatedDuration" AS expected_duration_minutes,
                   dep.depends_on_id
            FROM operations op
            LEFT JOIN work_centers wc ON wc.id = op."workCenterId"
            LEFT JOIN operation_dependencies dep ON dep.operation_id = op.id
            WHERE op.deleted_at IS NULL
            ORDER BY op.created_at, op.id
        """, tenant=tenant).to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(409, f"cannot read MO creation options: {exc}") from exc

    components_by_bom: dict[str, list[dict]] = {}
    for row in components:
        components_by_bom.setdefault(str(row["bom_id"]), []).append({
            "item_id": str(row["item_id"]) if not _is_missing(row.get("item_id")) else None,
            "name": row["name"],
            "required_quantity": float(row["required_quantity"]),
            "available_quantity": float(row["available_quantity"]),
        })
    operations_by_bom: dict[str, list[dict]] = {}
    for row in operations:
        operations_by_bom.setdefault(str(row["bom_id"]), []).append({
            "operation_id": str(row["id"]),
            "name": row["name"],
            "work_center_id": (
                str(row["work_center_id"])
                if not _is_missing(row.get("work_center_id")) else None
            ),
            "work_center_name": row.get("work_center_name") or "Work Center",
            "expected_duration_hours": float(row["expected_duration_minutes"]) / 60.0,
            "depends_on_operation_id": (
                str(row["depends_on_id"])
                if not _is_missing(row.get("depends_on_id")) else None
            ),
        })
    boms_by_product: dict[str, list[dict]] = {}
    for row in boms:
        bom_id = str(row["id"])
        boms_by_product.setdefault(str(row["product_id"]), []).append({
            "id": bom_id,
            "code": row["code"],
            "name": row["name"],
            "components": components_by_bom.get(bom_id, []),
            "operations": operations_by_bom.get(bom_id, []),
        })
    return {
        "tenant": tenant,
        "products": [
            {
                "id": str(row["id"]),
                "sku": row["sku"],
                "name": row["name"],
                "boms": boms_by_product.get(str(row["id"]), []),
            }
            for row in products
        ],
    }


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


class WorkOrderInput(BaseModel):
    operation_id: str | None = None
    work_center_id: str | None = None
    work_center_name: str = Field(default="Work Center", min_length=1, max_length=255)
    name: str = Field(min_length=1, max_length=255)
    expected_duration_hours: float = Field(gt=0, le=1000)
    actual_duration_hours: float | None = Field(default=None, ge=0, le=1000)
    units_done: float = Field(default=0, ge=0)
    depends_on_index: int | None = Field(default=None, ge=0)


class CreateMoRequest(BaseModel):
    product_id: str = Field(min_length=1)
    bom_id: str = Field(min_length=1)
    quantity: float = Field(gt=0)
    threshold: float = Field(default=1.0, gt=0)
    components: list[ComponentInput] = Field(default_factory=list, max_length=20)
    work_orders: list[WorkOrderInput] = Field(min_length=1, max_length=20)


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
            catalog = conn.execute(sa.text("""
                SELECT p.name AS product_name, b.name AS bom_name
                FROM products p JOIN boms b ON b.product_id = p.id
                WHERE p.id = CAST(:product AS uuid) AND b.id = CAST(:bom AS uuid)
                  AND p.deleted_at IS NULL AND b.deleted_at IS NULL
            """), {"product": body.product_id, "bom": body.bom_id}).mappings().first()
            if catalog is None:
                raise ValueError("selected BOM does not belong to the selected product")
            rows = [row.model_dump() for row in body.work_orders]
            for index, row in enumerate(rows):
                if row["units_done"] > body.quantity:
                    raise ValueError(
                        f"work order {index + 1} units done cannot exceed MO quantity"
                    )
                predecessor = row.get("depends_on_index")
                if predecessor is not None and predecessor >= index:
                    raise ValueError(
                        f"work order {index + 1} must depend on an earlier work order"
                    )
            insert_configured_job(
                conn,
                job_reference=job_reference,
                product_id=body.product_id,
                product_name=catalog["product_name"],
                bom_id=body.bom_id,
                bom_name=catalog["bom_name"],
                quantity=body.quantity,
                components=[c.model_dump() for c in body.components],
                work_orders=rows,
            )
    except Exception as exc:
        raise HTTPException(409, f"could not insert job: {exc}") from exc

    try:
        insights = run_delay_review(tenant=tenant, threshold=body.threshold, job_references=[job_reference])
    except Exception as exc:
        raise HTTPException(409, f"job was created but scoring failed: {exc}") from exc

    if not insights:
        raise HTTPException(409, "job was created but produced no scorable insight")

    orders = _load_manufacturing_orders(tenant, [job_reference])
    if len(orders) != 1:
        raise HTTPException(409, "job was scored but could not be reloaded from the database")
    return {
        "tenant": tenant,
        "job_id": job_reference,
        "insight": insights[0].to_dict(),
        "order": orders[0],
    }


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
