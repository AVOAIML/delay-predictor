"""End-to-end M3 pipeline test: Weight Agent weight generation ->
`ProductionDelayOrchestrator.resolve_risk_weights` (bp -> rule-engine
adapter) -> `calculate_delay_elements_for_jobs`. Run with `-s` to see the
printed resolved weights and enriched job output:

    uv run --extra dev pytest tests/rule_engine/test_m3_full_pipeline.py -q -s

Two scenarios are covered, matching the two ways `resolve_risk_weights` can
produce weights without a paid external LLM:
  - configured: tenant has its own configured_bp -> WeightAgent short-circuits
    to SOURCE_CONFIGURED, no LLM call at all.
  - cold start: no configured_bp, no history -> the two-stage LLM cold start
    (profile extraction, then weight adjustment), driven here by a scripted
    deterministic provider so the test never calls a paid API.
"""
from __future__ import annotations

import datetime as dt
import json

from m3_production_delay.llm_agents.weight_agent.models import SIGNAL_ORDER
from m3_production_delay.orchestrator import (
    ProductionDelayOrchestrator,
    WeightAgentRequest,
    _DemoScriptedLLMProvider,
)
from m3_production_delay.rule_engine.elements import calculate_delay_elements_for_jobs

ALL_AVAILABLE = {signal: True for signal in SIGNAL_ORDER}


def _sample_job_rollup() -> dict:
    """One job, two operations, op-b depends on op-a - same shape
    `build_job_rollups()` produces (see test_m3_elements.py's builders)."""
    operator = {
        "operator_id": "operator-1",
        "name": None,
        "last_10_work_orders": [
            {
                "work_order_id": "wo-hist",
                "elapsed_time_minutes": 500,
                "scheduled_time_minutes": 400,
                "completed_on": "2026-09-02",
                "operation_type": "INDEPENDENT",
            }
        ],
    }
    vendor = {
        "vendor_id": "vendor-1",
        "name": "Test Vendor",
        "last_10_purchase_orders": [
            {
                "po_number": "PO-1",
                "po_order_date": dt.datetime(2026, 8, 1),
                "po_order_deadline": dt.datetime(2026, 8, 15),
                "grn_received_date": dt.datetime(2026, 8, 22),
            }
        ],
    }
    component = {
        "component_id": "comp-1",
        "name": "Test Component",
        "required_quantity": 100,
        "available_quantity": 40,
        "vendor": vendor,
    }
    op_a = {
        "operation_id": "op-a",
        "operation_type": "INDEPENDENT",
        "status": "IN_PROGRESS",
        "depends_on_operation_ids": [],
        "expected_duration_minutes": 480,
        "actual_duration_minutes": 350,
        "job_quantity": 100.0,
        "current_done_quantity": 60.0,
        "operator_count": 1,
        "operators": [operator],
        "components": [component],
    }
    op_b = {
        "operation_id": "op-b",
        "operation_type": "DEPENDENT",
        "status": "NOT_STARTED",
        "depends_on_operation_ids": ["op-a"],
        "expected_duration_minutes": 300,
        "actual_duration_minutes": None,
        "job_quantity": 100.0,
        "current_done_quantity": 0.0,
        "operator_count": 1,
        "operators": [operator],
        "components": [component],
    }
    return {"job_id": "JOB-PIPELINE-TEST", "operations": [op_a, op_b]}


def _print_pipeline_result(scenario: str, risk_weights: dict, enriched_job: dict) -> None:
    print(f"\n=== M3 full pipeline ({scenario}) ===")
    print("resolved risk_weights:", json.dumps(risk_weights, indent=2))
    print("enriched job output:", json.dumps(enriched_job, indent=2, default=str))


def test_full_pipeline_configured_weights_through_elements():
    orchestrator = ProductionDelayOrchestrator()
    request = WeightAgentRequest(
        tenant_id="tenant-pipeline-configured",
        availability=ALL_AVAILABLE,
        configured_bp={
            "time_overrun": 4000,
            "operator_skill": 3000,
            "seasonality": 1000,
            "material_availability": 1500,
            "supplier_reliability": 500,
        },
    )

    risk_weights = orchestrator.resolve_risk_weights(request)
    assert risk_weights == {
        "time_overrun_ratio": 0.40,
        "operator_pace_ratio": 0.30,
        "material_shortfall_ratio": 0.15,
        "supplier_reliability": 0.05,
    }

    job_rollup = _sample_job_rollup()
    enriched_job = calculate_delay_elements_for_jobs([job_rollup], risk_weights=risk_weights)[0]
    _print_pipeline_result("configured", risk_weights, enriched_job)

    op_a, op_b = enriched_job["operations"]
    assert op_a["composite_risk_score"] is not None
    assert isinstance(op_a["is_delayed"], bool)
    assert op_b["predecessor_time_overrun_ratio"] == op_a["time_overrun_ratio"]


def test_full_pipeline_cold_start_llm_weights_through_elements():
    orchestrator = ProductionDelayOrchestrator(llm_provider=_DemoScriptedLLMProvider())
    request = WeightAgentRequest(
        tenant_id="tenant-pipeline-cold-start",
        availability=ALL_AVAILABLE,
        tenant_description=(
            "A make-to-order metal manufacturer with high dependency on imported "
            "raw materials and suppliers, stable workforce, low seasonal variation."
        ),
    )

    risk_weights = orchestrator.resolve_risk_weights(request)
    # No fixed expected values here (LLM-adjusted, not configured) - just the
    # invariants the adapter guarantees: the four rule-engine keys, no
    # `seasonality` leaking through. The four weights sum to 1.0 minus
    # whatever share `seasonality` took (dropped, not renormalized here -
    # `composite_risk_score` does its own renormalization downstream).
    assert set(risk_weights) == {
        "time_overrun_ratio",
        "operator_pace_ratio",
        "material_shortfall_ratio",
        "supplier_reliability",
    }
    assert 0.0 < sum(risk_weights.values()) <= 1.0

    job_rollup = _sample_job_rollup()
    enriched_job = calculate_delay_elements_for_jobs([job_rollup], risk_weights=risk_weights)[0]
    _print_pipeline_result("cold-start LLM", risk_weights, enriched_job)

    op_a, op_b = enriched_job["operations"]
    assert op_a["composite_risk_score"] is not None
    assert isinstance(op_a["is_delayed"], bool)
    assert op_b["predecessor_time_overrun_ratio"] == op_a["time_overrun_ratio"]
