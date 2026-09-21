"""Regenerates ``section1_job.json`` — a REAL
``rule_engine.elements.calculate_delay_elements_for_jobs()`` output, recorded
so the review tests run against the shape the Risk Engine actually produces
rather than a hand-written approximation of it.

Run from the repo root:

    uv run python modules/m3_production_delay/review/fixtures/build_fixture.py

The job it builds is deliberately awkward, because the awkward cases are the
ones Section 3 has to get right:

  * ``Cutting`` — well past the scoring gate and genuinely over its planned
    time, with an operator whose history runs long.
  * ``Welding`` — 30% elapsed, so it clears the 25% gate but is not yet over
    its plan: a scorable operation with an unfired time signal.
  * ``Assembly`` — not started, depends on ``Cutting``: not scorable, and a
    cascading-predecessor context flag.

All three carry the identical component list, because ``rollup.py`` attaches
the manufacturing order's components to every one of its operations — the
duplication the job-scoped material and supplier evidence has to collapse.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from m3_production_delay.rule_engine.elements import (
    DEFAULT_DELAY_THRESHOLD,
    calculate_delay_elements_for_jobs,
)

OUT_PATH = Path(__file__).resolve().parent / "section1_job.json"

#: The same shape the review tests score against — a resolved tenant vector,
#: which never sums to 1.0 (the Weight Agent's `seasonality` share is dropped).
WEIGHTS = {
    "time_overrun_ratio": 0.40,
    "operator_pace_ratio": 0.30,
    "material_shortfall_ratio": 0.15,
    "supplier_reliability": 0.05,
}


def _operator(token: str, elapsed: int, scheduled: int) -> dict:
    return {
        "operator_id": token,  # HMAC token, never a raw uuid — name stays None
        "name": None,
        "last_10_work_orders": [
            {
                "work_order_id": "wo-hist-1",
                "elapsed_time_minutes": elapsed,
                "scheduled_time_minutes": scheduled,
                "completed_on": dt.datetime(2026, 9, 2),
                "operation_type": "INDEPENDENT",
            }
        ],
    }


def _components() -> list[dict]:
    late_vendor = {
        "vendor_id": "ven-steel",
        "name": "Lanka Steel",
        "last_10_purchase_orders": [
            {
                "po_number": "PO-8801",
                "po_order_date": dt.datetime(2026, 8, 1),
                "po_order_deadline": dt.datetime(2026, 8, 15),
                "grn_received_date": dt.datetime(2026, 8, 22),
            }
        ],
    }
    on_time_vendor = {
        "vendor_id": "ven-fast",
        "name": "Colombo Fasteners",
        "last_10_purchase_orders": [
            {
                "po_number": "PO-8802",
                "po_order_date": dt.datetime(2026, 8, 1),
                "po_order_deadline": dt.datetime(2026, 8, 15),
                "grn_received_date": dt.datetime(2026, 8, 12),
            }
        ],
    }
    return [
        {
            "component_id": "item-steel-plate",
            "name": "Steel Plate 12mm",
            "required_quantity": 100.0,
            "available_quantity": 40.0,
            "vendor": late_vendor,
        },
        {
            "component_id": "item-bolt-m8",
            "name": "Bolt M8",
            "required_quantity": 500.0,
            "available_quantity": 500.0,  # not short
            "vendor": on_time_vendor,
        },
    ]


def build_job() -> dict:
    components = _components()
    slow_operator = _operator("hmac-operator-a", elapsed=500, scheduled=400)  # pace 1.25
    steady_operator = _operator("hmac-operator-b", elapsed=380, scheduled=400)  # pace 0.95

    cutting = {
        "operation_id": "wo-cutting-0001",
        "operation_type": "INDEPENDENT",
        "status": "IN_PROGRESS",
        "depends_on_operation_ids": [],
        "expected_duration_minutes": 480,
        "actual_duration_minutes": 600,
        "job_quantity": 100.0,
        "current_done_quantity": 60.0,
        "operator_count": 1,
        "operators": [slow_operator],
        "components": components,
    }
    welding = {
        "operation_id": "wo-welding-0002",
        "operation_type": "INDEPENDENT",
        "status": "IN_PROGRESS",
        "depends_on_operation_ids": [],
        "expected_duration_minutes": 300,
        "actual_duration_minutes": 90,  # 30% elapsed: scorable, not overrunning
        "job_quantity": 100.0,
        "current_done_quantity": 35.0,
        "operator_count": 1,
        "operators": [steady_operator],
        "components": components,
    }
    assembly = {
        "operation_id": "wo-assembly-0003",
        "operation_type": "DEPENDENT",
        "status": "NOT_STARTED",
        "depends_on_operation_ids": ["wo-cutting-0001"],
        "expected_duration_minutes": 240,
        "actual_duration_minutes": None,
        "job_quantity": 100.0,
        "current_done_quantity": 0.0,
        "operator_count": 1,
        "operators": [slow_operator],
        "components": components,
    }
    rollup = {"job_id": "WH/MO/00142", "operations": [cutting, welding, assembly]}
    return calculate_delay_elements_for_jobs(
        [rollup], risk_weights=WEIGHTS, delay_threshold=DEFAULT_DELAY_THRESHOLD
    )[0]


def main() -> None:
    OUT_PATH.write_text(
        json.dumps(build_job(), indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
