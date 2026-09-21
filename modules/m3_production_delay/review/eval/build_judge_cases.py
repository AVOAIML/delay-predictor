"""Regenerates ``judge_cases.json`` — the judge's evaluation set.

Twenty cases over four real evidence packs: ten whose candidate lines are
fully supported by the evidence, and ten that are not, split across the four
ways a wrong explanation actually shows up:

  * ``fabricated_cause``  — a cause no signal in this system can observe.
  * ``wrong_number``      — a real cause quoting a number the evidence denies.
  * ``non_fired_signal``  — a real signal cited on an operation where it did
                            not clear its baseline.
  * ``omitted_signal``    — every line true, but a fired weighted cause is
                            missing from the set.

The evidence packs are built by the real ``build_evidence`` over real
``calculate_delay_elements_for_jobs`` output, and the supported lines by the
real composer, so the set cannot drift from what the pipeline actually sends.
Only the unsupported variants are hand-written — that is the whole point.

Run from the repo root:

    uv run python modules/m3_production_delay/review/eval/build_judge_cases.py
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from pathlib import Path

from maxxflow_core.jsonutil import json_default

from m3_production_delay.review.composer import compose
from m3_production_delay.review.evidence import build_evidence
from m3_production_delay.review.fixtures.build_fixture import build_job
from m3_production_delay.review.schemas import (
    SIGNAL_MATERIAL_SHORTFALL,
    SIGNAL_OPERATOR_PACE,
    SIGNAL_SUPPLIER_RELIABILITY,
    SIGNAL_TIME_OVERRUN,
)
from m3_production_delay.rule_engine.elements import calculate_delay_elements_for_jobs

OUT_PATH = Path(__file__).resolve().parent / "judge_cases.json"

FULL_WEIGHTS = {
    "time_overrun_ratio": 0.40,
    "operator_pace_ratio": 0.30,
    "material_shortfall_ratio": 0.15,
    "supplier_reliability": 0.05,
}
TIME_ONLY_WEIGHTS = {**FULL_WEIGHTS, "material_shortfall_ratio": 0.0, "supplier_reliability": 0.0}
THRESHOLD = 1.0


def _material_only_job() -> dict:
    """A job whose time and operator signals are quiet, so only the two
    job-scoped causes fire — the mirror image of the time-only scenario."""
    component = {
        "component_id": "item-resin",
        "name": "Epoxy Resin",
        "required_quantity": 200.0,
        "available_quantity": 75.0,
        "vendor": {
            "vendor_id": "ven-chem",
            "name": "Ceylon Chemicals",
            "last_10_purchase_orders": [
                {
                    "po_number": "PO-9001",
                    "po_order_date": dt.datetime(2026, 8, 1),
                    "po_order_deadline": dt.datetime(2026, 8, 11),
                    "grn_received_date": dt.datetime(2026, 8, 21),
                }
            ],
        },
    }
    steady_operator = {
        "operator_id": "hmac-operator-c",
        "name": None,
        "last_10_work_orders": [
            {
                "work_order_id": "wo-hist-9",
                "elapsed_time_minutes": 380,
                "scheduled_time_minutes": 400,
                "completed_on": dt.datetime(2026, 9, 1),
                "operation_type": "INDEPENDENT",
            }
        ],
    }
    moulding = {
        "operation_id": "wo-moulding-0001",
        "operation_type": "INDEPENDENT",
        "status": "IN_PROGRESS",
        "depends_on_operation_ids": [],
        "expected_duration_minutes": 600,
        "actual_duration_minutes": 300,  # half elapsed, on schedule
        "job_quantity": 200.0,
        "current_done_quantity": 110.0,
        "operator_count": 1,
        "operators": [steady_operator],
        "components": [component],
    }
    return calculate_delay_elements_for_jobs(
        [{"job_id": "WH/MO/00420", "operations": [moulding]}],
        risk_weights=FULL_WEIGHTS,
        delay_threshold=THRESHOLD,
    )[0]


def _simple_time_job() -> dict:
    """One operation, one cause: time. Nothing else has any evidence at all."""
    op = {
        "operation_id": "wo-press-0001",
        "operation_name": "Pressing",
        "operation_type": "INDEPENDENT",
        "status": "IN_PROGRESS",
        "depends_on_operation_ids": [],
        "expected_duration_minutes": 240,
        "actual_duration_minutes": 360,
        "job_quantity": 50.0,
        "current_done_quantity": 30.0,
        "operator_count": 0,
        "operators": [],
        "components": [],
    }
    return calculate_delay_elements_for_jobs(
        [{"job_id": "WH/MO/00430", "operations": [op]}],
        risk_weights=FULL_WEIGHTS,
        delay_threshold=THRESHOLD,
    )[0]


def _scenarios() -> dict[str, dict]:
    return {
        # The recorded fixture: three operations, two scorable, all four
        # causes firing.
        "full_job": build_evidence(build_job(), FULL_WEIGHTS, THRESHOLD),
        # The same job for a tenant that weights material and supplier at
        # zero: the identical evidence must produce a shorter, still-complete
        # explanation.
        "time_only_weights": build_evidence(build_job(), TIME_ONLY_WEIGHTS, THRESHOLD),
        # Only the two job-scoped causes fire.
        "material_job": build_evidence(_material_only_job(), FULL_WEIGHTS, THRESHOLD),
        # A single operation with a single cause.
        "simple_time_job": build_evidence(_simple_time_job(), FULL_WEIGHTS, THRESHOLD),
    }


def _line_index(draft, signal_key: str) -> int:
    return next(i for i, line in enumerate(draft.why_lines) if line.signal_key == signal_key)


def _lines(draft) -> list[dict]:
    return [line.to_dict() for line in draft.why_lines]


def _mutated(draft, signal_key: str, **changes) -> list[dict]:
    lines = list(draft.why_lines)
    index = _line_index(draft, signal_key)
    lines[index] = dataclasses.replace(lines[index], **changes)
    return [line.to_dict() for line in draft.with_lines(tuple(lines)).why_lines]


def _without(draft, signal_key: str) -> list[dict]:
    kept = tuple(line for line in draft.why_lines if line.signal_key != signal_key)
    return [line.to_dict() for line in draft.with_lines(kept).why_lines]


def build_cases() -> dict:
    scenarios = _scenarios()
    drafts = {name: compose(pack) for name, pack in scenarios.items()}
    full = drafts["full_job"]
    time_only = drafts["time_only_weights"]
    material = drafts["material_job"]
    simple = drafts["simple_time_job"]

    def reordered(draft) -> list[dict]:
        lines = list(draft.why_lines)
        return [line.to_dict() for line in draft.with_lines(tuple(reversed(lines))).why_lines]

    cases: list[dict] = [
        # ── supported ────────────────────────────────────────────────────
        {
            "id": "supported-full-set",
            "category": "supported",
            "scenario": "full_job",
            "lines": _lines(full),
            "target_index": _line_index(full, SIGNAL_TIME_OVERRUN),
            "expected_supported": True,
            "expected_approved": True,
            "note": "the composer's own output over the recorded fixture",
        },
        {
            "id": "supported-full-set-reordered",
            "category": "supported",
            "scenario": "full_job",
            "lines": reordered(full),
            "target_index": 0,
            "expected_supported": True,
            "expected_approved": True,
            "note": "order is a presentation choice, not a support question",
        },
        {
            "id": "supported-operator-line",
            "category": "supported",
            "scenario": "full_job",
            "lines": _lines(full),
            "target_index": _line_index(full, SIGNAL_OPERATOR_PACE),
            "expected_supported": True,
            "expected_approved": True,
            "note": "operator pace 1.25 against a 1.2 baseline",
        },
        {
            "id": "supported-material-line",
            "category": "supported",
            "scenario": "full_job",
            "lines": _lines(full),
            "target_index": _line_index(full, SIGNAL_MATERIAL_SHORTFALL),
            "expected_supported": True,
            "expected_approved": True,
            "note": "60 short of 100 required",
        },
        {
            "id": "supported-supplier-line",
            "category": "supported",
            "scenario": "full_job",
            "lines": _lines(full),
            "target_index": _line_index(full, SIGNAL_SUPPLIER_RELIABILITY),
            "expected_supported": True,
            "expected_approved": True,
            "note": "mean lead-time ratio above 1.0 across the named vendors",
        },
        {
            "id": "supported-zero-weighted-signals-absent",
            "category": "supported",
            "scenario": "time_only_weights",
            "lines": _lines(time_only),
            "target_index": _line_index(time_only, SIGNAL_TIME_OVERRUN),
            "expected_supported": True,
            "expected_approved": True,
            "note": (
                "material and supplier fired but carry zero weight for this tenant, so "
                "their absence is correct, not an omission"
            ),
        },
        {
            "id": "supported-zero-weight-reordered",
            "category": "supported",
            "scenario": "time_only_weights",
            "lines": reordered(time_only),
            "target_index": 0,
            "expected_supported": True,
            "expected_approved": True,
            "note": "same shorter set, different order",
        },
        {
            "id": "supported-single-cause",
            "category": "supported",
            "scenario": "simple_time_job",
            "lines": _lines(simple),
            "target_index": 0,
            "expected_supported": True,
            "expected_approved": True,
            "note": "one operation, one fired signal, one line",
        },
        {
            "id": "supported-material-only-job",
            "category": "supported",
            "scenario": "material_job",
            "lines": _lines(material),
            "target_index": _line_index(material, SIGNAL_MATERIAL_SHORTFALL),
            "expected_supported": True,
            "expected_approved": True,
            "note": "time and operator are quiet; only the job-scoped causes fire",
        },
        {
            "id": "supported-material-only-supplier",
            "category": "supported",
            "scenario": "material_job",
            "lines": _lines(material),
            "target_index": _line_index(material, SIGNAL_SUPPLIER_RELIABILITY),
            "expected_supported": True,
            "expected_approved": True,
            "note": "a genuinely late vendor on the short component",
        },
        # ── fabricated cause ─────────────────────────────────────────────
        {
            "id": "unsupported-fabricated-breakdown",
            "category": "fabricated_cause",
            "scenario": "full_job",
            "lines": _mutated(
                full,
                SIGNAL_TIME_OVERRUN,
                headline="Machine breakdown on the cutting line caused the overrun",
            ),
            "target_index": _line_index(full, SIGNAL_TIME_OVERRUN),
            "expected_supported": False,
            "expected_approved": False,
            "note": "no signal observes machine downtime",
        },
        {
            "id": "unsupported-fabricated-inspection",
            "category": "fabricated_cause",
            "scenario": "full_job",
            "lines": _mutated(
                full,
                SIGNAL_OPERATOR_PACE,
                headline="Operator was waiting on a quality inspection",
            ),
            "target_index": _line_index(full, SIGNAL_OPERATOR_PACE),
            "expected_supported": False,
            "expected_approved": False,
            "note": "pace history says nothing about what the operator was waiting for",
        },
        {
            "id": "unsupported-fabricated-weather",
            "category": "fabricated_cause",
            "scenario": "material_job",
            "lines": _mutated(
                material,
                SIGNAL_SUPPLIER_RELIABILITY,
                headline="Supplier shipments were held up by bad weather",
            ),
            "target_index": _line_index(material, SIGNAL_SUPPLIER_RELIABILITY),
            "expected_supported": False,
            "expected_approved": False,
            "note": "lead-time ratios carry no reason for lateness",
        },
        # ── wrong number ─────────────────────────────────────────────────
        {
            "id": "unsupported-wrong-hours",
            "category": "wrong_number",
            "scenario": "full_job",
            "lines": _mutated(
                full,
                SIGNAL_TIME_OVERRUN,
                detail="INDEPENDENT · wo-cutti · 26.0 hrs logged of 8.0 planned",
                delta="+18.0 hrs",
                quoted={"actual_hrs": 26.0, "expected_hrs": 8.0, "delta_hrs": 18.0},
            ),
            "target_index": _line_index(full, SIGNAL_TIME_OVERRUN),
            "expected_supported": False,
            "expected_approved": False,
            "note": "the evidence says 10.0 hrs against 8.0",
        },
        {
            "id": "unsupported-wrong-pace",
            "category": "wrong_number",
            "scenario": "full_job",
            "lines": _mutated(
                full,
                SIGNAL_OPERATOR_PACE,
                detail=(
                    "INDEPENDENT · wo-cutti · operator pace 3.40× planned over "
                    "recent completed jobs"
                ),
                quoted={"pace_ratio": 3.4},
            ),
            "target_index": _line_index(full, SIGNAL_OPERATOR_PACE),
            "expected_supported": False,
            "expected_approved": False,
            "note": "the evidence says 1.25",
        },
        {
            "id": "unsupported-wrong-shortfall",
            "category": "wrong_number",
            "scenario": "full_job",
            "lines": _mutated(
                full,
                SIGNAL_MATERIAL_SHORTFALL,
                detail="WH/MO/00142 · Steel Plate 12mm – 480 short",
                quoted={"shortfall:item-steel-plate": 480.0},
            ),
            "target_index": _line_index(full, SIGNAL_MATERIAL_SHORTFALL),
            "expected_supported": False,
            "expected_approved": False,
            "note": "100 required against 40 available is 60 short, not 480",
        },
        # ── non-fired signal ─────────────────────────────────────────────
        {
            "id": "unsupported-time-on-a-quiet-operation",
            "category": "non_fired_signal",
            "scenario": "full_job",
            "lines": _mutated(full, SIGNAL_TIME_OVERRUN, operation_id="wo-welding-0002"),
            "target_index": _line_index(full, SIGNAL_TIME_OVERRUN),
            "expected_supported": False,
            "expected_approved": False,
            "note": "welding is 30% elapsed and not overrunning; its time signal did not fire",
        },
        {
            "id": "unsupported-operator-that-did-not-fire",
            "category": "non_fired_signal",
            "scenario": "material_job",
            "lines": _lines(material)
            + [
                {
                    "index": len(material.why_lines),
                    "signal_key": SIGNAL_OPERATOR_PACE,
                    "scope": "operation",
                    "headline": "Assigned operator has a history of overrunning",
                    "detail": (
                        "INDEPENDENT · wo-mouldi · operator pace 0.95× planned "
                        "over recent completed jobs"
                    ),
                    "delta": None,
                    "operation_id": "wo-moulding-0001",
                    "contribution": None,
                    "quoted": {"pace_ratio": 0.95},
                }
            ],
            "target_index": len(material.why_lines),
            "expected_supported": False,
            "expected_approved": False,
            "note": "0.95 is faster than planned, and below the 1.2 baseline either way",
        },
        # ── omitted signal ───────────────────────────────────────────────
        {
            "id": "unsupported-omits-operator",
            "category": "omitted_signal",
            "scenario": "full_job",
            "lines": _without(full, SIGNAL_OPERATOR_PACE),
            "target_index": None,
            "expected_supported": None,
            "expected_approved": False,
            "note": "every line true, but a fired weighted cause is missing",
        },
        {
            "id": "unsupported-omits-material",
            "category": "omitted_signal",
            "scenario": "full_job",
            "lines": _without(full, SIGNAL_MATERIAL_SHORTFALL),
            "target_index": None,
            "expected_supported": None,
            "expected_approved": False,
            "note": "the largest shortage in the job is not mentioned at all",
        },
    ]

    return {
        "version": 1,
        "threshold": THRESHOLD,
        "scenarios": {name: pack.to_dict() for name, pack in _scenarios().items()},
        "cases": cases,
    }


def main() -> None:
    payload = build_cases()
    supported = sum(1 for case in payload["cases"] if case["expected_approved"])
    OUT_PATH.write_text(
        json.dumps(payload, indent=2, default=json_default, allow_nan=False, sort_keys=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {OUT_PATH} — {len(payload['cases'])} cases "
        f"({supported} supported, {len(payload['cases']) - supported} unsupported)"
    )


if __name__ == "__main__":
    main()
