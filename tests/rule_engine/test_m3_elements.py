"""Unit tests for m3_delay.elements - pure functions over the dict shape
build_job_rollups() produces, no DB/Postgres needed at all (unlike
tests/unit/test_m3_rollup_db.py, which is needs_db). Every function is
exercised individually plus calculate_delay_elements end-to-end across
several realistic job scenarios.
"""
from __future__ import annotations

import math

import pytest

from m3_production_delay.rule_engine.elements import (
    calculate_delay_elements_for_jobs,
    composite_risk_score,
    is_delayed,
    material_shortfall_ratio,
    operator_pace_ratio,
    predecessor_time_overrun_ratio,
    predicted_overrun_hours,
    time_overrun_ratio,
    vendor_lead_time_ratio,
    weights_bp_to_risk_weights,
    DEFAULT_RISK_WEIGHTS,
)


# ─── small builders so each test only states what it actually varies ───────


def make_op(**overrides) -> dict:
    op = {
        "operation_id": "op-1",
        "operation_type": "INDEPENDENT",
        "status": "IN_PROGRESS",
        "depends_on_operation_ids": [],
        "expected_duration_minutes": 480,
        "actual_duration_minutes": 350,
        "job_quantity": 100.0,
        "current_done_quantity": 60.0,
        "operator_count": 0,
        "operators": [],
        "components": [],
    }
    op.update(overrides)
    return op


def make_operator(work_orders: list[dict]) -> dict:
    return {"operator_id": "op-token", "name": None, "last_10_work_orders": work_orders}


def make_history_entry(
    elapsed: float, scheduled: float, completed_on="2026-01-01", **overrides,
) -> dict:
    entry = {
        "work_order_id": "wo-hist",
        "elapsed_time_minutes": elapsed,
        "scheduled_time_minutes": scheduled,
        "completed_on": completed_on,
        "operation_type": "INDEPENDENT",
    }
    entry.update(overrides)
    return entry


def make_component(required: float, available: float, vendor: dict | None = None) -> dict:
    return {
        "component_id": "comp-1",
        "name": "Test Component",
        "required_quantity": required,
        "available_quantity": available,
        "vendor": vendor,
    }


def make_vendor(purchase_orders: list[dict]) -> dict:
    return {"vendor_id": "ven-1", "name": "Test Vendor", "last_10_purchase_orders": purchase_orders}


def make_po(order_date, deadline, received, po_number="PO-1") -> dict:
    return {
        "po_number": po_number,
        "po_order_date": order_date,
        "po_order_deadline": deadline,
        "grn_received_date": received,
    }


# ─── time_overrun_ratio ─────────────────────────────────────────────────────


def test_time_overrun_ratio_normal():
    op = make_op(actual_duration_minutes=350, expected_duration_minutes=480)
    assert time_overrun_ratio(op) == pytest.approx(350 / 480)


def test_time_overrun_ratio_none_when_not_started():
    op = make_op(actual_duration_minutes=None)
    assert time_overrun_ratio(op) is None


def test_time_overrun_ratio_none_when_expected_zero():
    op = make_op(actual_duration_minutes=100, expected_duration_minutes=0)
    assert time_overrun_ratio(op) is None


def test_time_overrun_ratio_none_when_actual_is_zero():
    # 0 logged minutes is not a real "0% overrun" - treated like not-started.
    op = make_op(actual_duration_minutes=0, expected_duration_minutes=480)
    assert time_overrun_ratio(op) is None


# ─── predicted_overrun_hours ────────────────────────────────────────────────


def test_predicted_overrun_hours_quantity_based():
    op = make_op(
        actual_duration_minutes=350, expected_duration_minutes=480,
        current_done_quantity=60.0, job_quantity=100.0,
    )
    predicted_total = 350 * (100 / 60)
    assert predicted_overrun_hours(op) == pytest.approx((predicted_total - 480) / 60.0)


def test_predicted_overrun_hours_falls_back_to_operator_history_when_no_quantity_progress():
    # not started at all, but the assigned operator has a real pace history
    op = make_op(
        actual_duration_minutes=None,
        current_done_quantity=0.0,
        expected_duration_minutes=300,
        operators=[make_operator([make_history_entry(elapsed=500, scheduled=400)])],
    )
    # operator_pace_ratio = 500/400 = 1.25 -> predicted_total = 300*1.25 = 375
    assert predicted_overrun_hours(op) == pytest.approx((375 - 300) / 60.0)


def test_predicted_overrun_hours_none_when_no_basis_at_all():
    op = make_op(actual_duration_minutes=None, current_done_quantity=0.0, operators=[])
    assert predicted_overrun_hours(op) is None


def test_predicted_overrun_hours_falls_back_when_actual_is_zero():
    # actual_duration_minutes=0 must NOT be treated as real quantity-based
    # progress (0 * anything = 0, a meaningless "instant" projection) - it
    # should fall through to the operator-history basis, same as actual=None.
    op = make_op(
        actual_duration_minutes=0,
        current_done_quantity=60.0, job_quantity=100.0,  # progress values ARE present...
        expected_duration_minutes=300,
        operators=[make_operator([make_history_entry(elapsed=500, scheduled=400)])],
    )
    assert predicted_overrun_hours(op) == pytest.approx((300 * 1.25 - 300) / 60.0)


def test_predicted_overrun_hours_none_when_expected_missing():
    op = make_op(expected_duration_minutes=None)
    assert predicted_overrun_hours(op) is None


# ─── predecessor_time_overrun_ratio ─────────────────────────────────────────


def test_predecessor_time_overrun_ratio_none_for_independent_operation():
    op = make_op(depends_on_operation_ids=[])
    assert predecessor_time_overrun_ratio(op, {}) is None


def test_predecessor_time_overrun_ratio_takes_max_of_known_predecessors():
    op = make_op(depends_on_operation_ids=["pred-a", "pred-b"])
    ratios_by_id = {"pred-a": 0.8, "pred-b": 1.4}
    assert predecessor_time_overrun_ratio(op, ratios_by_id) == 1.4


def test_predecessor_time_overrun_ratio_excludes_unknown_predecessors():
    op = make_op(depends_on_operation_ids=["pred-a", "pred-unknown"])
    ratios_by_id = {"pred-a": 0.9}  # pred-unknown not in map at all
    assert predecessor_time_overrun_ratio(op, ratios_by_id) == 0.9


def test_predecessor_time_overrun_ratio_none_when_all_predecessors_unresolved():
    op = make_op(depends_on_operation_ids=["pred-a"])
    ratios_by_id = {"pred-a": None}  # predecessor exists but has no ratio itself
    assert predecessor_time_overrun_ratio(op, ratios_by_id) is None


# ─── operator_pace_ratio ─────────────────────────────────────────────────────


def test_operator_pace_ratio_averages_only_completed_work_orders():
    operator = make_operator([
        make_history_entry(elapsed=110, scheduled=100, completed_on="2026-01-01"),
        make_history_entry(elapsed=999, scheduled=100, completed_on=None),  # in progress - excluded
        make_history_entry(elapsed=90, scheduled=100, completed_on="2026-01-02"),
    ])
    op = make_op(operators=[operator])
    assert operator_pace_ratio(op) == pytest.approx((1.10 + 0.90) / 2)


def test_operator_pace_ratio_averages_across_multiple_operators():
    op = make_op(operators=[
        make_operator([make_history_entry(elapsed=120, scheduled=100)]),  # R=1.2
        make_operator([make_history_entry(elapsed=90, scheduled=100)]),   # R=0.9
    ])
    assert operator_pace_ratio(op) == pytest.approx((1.2 + 0.9) / 2)


def test_operator_pace_ratio_none_when_no_operators():
    op = make_op(operators=[])
    assert operator_pace_ratio(op) is None


def test_operator_pace_ratio_excludes_operator_with_zero_completed_history():
    op = make_op(operators=[
        make_operator([make_history_entry(elapsed=999, scheduled=100, completed_on=None)]),  # all in-progress
        make_operator([make_history_entry(elapsed=150, scheduled=100)]),  # R=1.5
    ])
    # the first operator contributes NOTHING (not a 0, not skipped-as-1.0)
    assert operator_pace_ratio(op) == pytest.approx(1.5)


def test_operator_pace_ratio_guards_zero_scheduled_time():
    operator = make_operator([make_history_entry(elapsed=50, scheduled=0)])
    op = make_op(operators=[operator])
    assert operator_pace_ratio(op) is None


# ─── material_shortfall_ratio ────────────────────────────────────────────────


def test_material_shortfall_ratio_zero_when_no_components():
    assert material_shortfall_ratio(make_op(components=[])) == 0.0


def test_material_shortfall_ratio_zero_when_no_shortage():
    op = make_op(components=[make_component(required=50, available=100)])
    assert material_shortfall_ratio(op) == 0.0


def test_material_shortfall_ratio_single_shortage():
    op = make_op(components=[make_component(required=100, available=40)])
    assert material_shortfall_ratio(op) == pytest.approx(100 / 40)


def test_material_shortfall_ratio_infinite_when_available_is_zero():
    op = make_op(components=[make_component(required=100, available=0)])
    assert material_shortfall_ratio(op) == math.inf


def test_material_shortfall_ratio_sums_multiple_shortages():
    op = make_op(components=[
        make_component(required=100, available=40),  # 2.5
        make_component(required=20, available=100),  # no shortage - 0
        make_component(required=10, available=5),    # 2.0
    ])
    assert material_shortfall_ratio(op) == pytest.approx(2.5 + 2.0)


# ─── vendor_lead_time_ratio ──────────────────────────────────────────────────


def test_vendor_lead_time_ratio_none_when_no_vendor():
    assert vendor_lead_time_ratio(None) is None


def test_vendor_lead_time_ratio_late_delivery():
    import datetime as dt
    vendor = make_vendor([make_po(
        order_date=dt.datetime(2026, 8, 1),
        deadline=dt.datetime(2026, 8, 15),   # 14-day window
        received=dt.datetime(2026, 8, 22),   # 21 days actual
    )])
    assert vendor_lead_time_ratio(vendor) == pytest.approx(21 / 14)


def test_vendor_lead_time_ratio_excludes_po_missing_a_date():
    import datetime as dt
    vendor = make_vendor([
        make_po(dt.datetime(2026, 8, 1), dt.datetime(2026, 8, 15), None),  # not yet received
        make_po(dt.datetime(2026, 7, 1), dt.datetime(2026, 7, 15), dt.datetime(2026, 7, 15)),  # on time, ratio=1.0
    ])
    assert vendor_lead_time_ratio(vendor) == pytest.approx(1.0)


def test_vendor_lead_time_ratio_excludes_zero_length_window():
    import datetime as dt
    same_day = dt.datetime(2026, 8, 1)
    vendor = make_vendor([make_po(same_day, same_day, dt.datetime(2026, 8, 3))])
    assert vendor_lead_time_ratio(vendor) is None


def test_vendor_lead_time_ratio_none_when_no_purchase_orders():
    assert vendor_lead_time_ratio(make_vendor([])) is None


# ─── weights_bp_to_risk_weights ──────────────────────────────────────────────


def test_weights_bp_to_risk_weights_maps_names_and_scales_to_fraction():
    weights_bp = {
        "time_overrun": 4000,
        "operator_skill": 3000,
        "seasonality": 1000,
        "material_availability": 1500,
        "supplier_reliability": 500,
    }
    assert weights_bp_to_risk_weights(weights_bp) == {
        "time_overrun_ratio": 0.40,
        "operator_pace_ratio": 0.30,
        "material_shortfall_ratio": 0.15,
        "supplier_reliability": 0.05,
    }


def test_weights_bp_to_risk_weights_drops_seasonality():
    weights_bp = {
        "time_overrun": 2000,
        "operator_skill": 2000,
        "seasonality": 6000,
        "material_availability": 0,
        "supplier_reliability": 0,
    }
    assert "seasonality" not in weights_bp_to_risk_weights(weights_bp)


def test_weights_bp_to_risk_weights_feeds_composite_risk_score_normalized():
    weights_bp = {
        "time_overrun": 4000,
        "operator_skill": 3000,
        "seasonality": 1000,
        "material_availability": 1500,
        "supplier_reliability": 500,
    }
    risk_weights = weights_bp_to_risk_weights(weights_bp)
    score = composite_risk_score(
        {
            "time_overrun_ratio": 2.0,
            "operator_pace_ratio": 2.0,
            "material_shortfall_ratio": 2.0,
            "supplier_reliability": 2.0,
        },
        risk_weights,
    )
    # weights already sum to 1.0 (10000bp / 10000), and every value is the
    # same 2.0, so the weighted average collapses to exactly 2.0 regardless
    # of how the weight is distributed across the four keys.
    assert score == pytest.approx(2.0)


# ─── composite_risk_score ────────────────────────────────────────────────────


def test_composite_risk_score_full_weighted_average():
    values = {
        "time_overrun_ratio": 1.0,
        "operator_pace_ratio": 1.2,
        "material_shortfall_ratio": 2.0,
        "supplier_reliability": 1.5,
    }
    expected = (0.50 * 1.0) + (0.35 * 1.2) + (0.10 * 2.0) + (0.05 * 1.5)
    assert composite_risk_score(values, DEFAULT_RISK_WEIGHTS) == pytest.approx(expected)


def test_composite_risk_score_renormalizes_when_a_value_is_missing():
    values = {
        "time_overrun_ratio": None,  # missing - its 50% weight must be excluded, not treated as 0
        "operator_pace_ratio": 1.2,
        "material_shortfall_ratio": 2.0,
        "supplier_reliability": 1.5,
    }
    remaining_weight = 0.35 + 0.10 + 0.05
    expected = ((0.35 * 1.2) + (0.10 * 2.0) + (0.05 * 1.5)) / remaining_weight
    assert composite_risk_score(values, DEFAULT_RISK_WEIGHTS) == pytest.approx(expected)


def test_composite_risk_score_none_when_everything_missing():
    values = {k: None for k in DEFAULT_RISK_WEIGHTS}
    assert composite_risk_score(values, DEFAULT_RISK_WEIGHTS) is None


def test_composite_risk_score_propagates_infinite_material_shortfall():
    values = {
        "time_overrun_ratio": 1.0,
        "operator_pace_ratio": 1.0,
        "material_shortfall_ratio": math.inf,
        "supplier_reliability": 1.0,
    }
    assert composite_risk_score(values, DEFAULT_RISK_WEIGHTS) == math.inf


def test_composite_risk_score_custom_weights():
    values = {"a": 2.0, "b": 4.0}
    weights = {"a": 0.25, "b": 0.75}
    assert composite_risk_score(values, weights) == pytest.approx(0.25 * 2.0 + 0.75 * 4.0)


# ─── is_delayed ──────────────────────────────────────────────────────────────


def test_is_delayed_above_threshold():
    assert is_delayed(1.5, threshold=1.0) is True


def test_is_delayed_at_threshold_is_false():
    assert is_delayed(1.0, threshold=1.0) is False


def test_is_delayed_below_threshold():
    assert is_delayed(0.5, threshold=1.0) is False


def test_is_delayed_none_when_score_is_none():
    assert is_delayed(None, threshold=1.0) is None


# ─── calculate_delay_elements: end-to-end scenarios ─────────────────────────


def _rollup_with_two_dependent_operations() -> dict:
    """op-b depends on op-a. Same shape as the real-DB scenario in
    scripts/run_elements_real_db.py, but as a plain dict (no DB round trip)."""
    operator = make_operator([make_history_entry(elapsed=500, scheduled=400, completed_on="2026-09-02")])

    import datetime as dt
    vendor = make_vendor([make_po(
        dt.datetime(2026, 8, 1), dt.datetime(2026, 8, 15), dt.datetime(2026, 8, 22),
    )])
    component = make_component(required=100, available=40, vendor=vendor)

    op_a = make_op(
        operation_id="op-a",
        depends_on_operation_ids=[],
        actual_duration_minutes=350,
        expected_duration_minutes=480,
        current_done_quantity=60.0,
        job_quantity=100.0,
        operators=[operator],
        components=[component],
    )
    op_b = make_op(
        operation_id="op-b",
        depends_on_operation_ids=["op-a"],
        actual_duration_minutes=None,
        expected_duration_minutes=300,
        current_done_quantity=0.0,
        job_quantity=100.0,
        operators=[operator],
        components=[component],
    )
    return {"job_id": "JOB-TEST-1", "operations": [op_a, op_b]}


def test_calculate_delay_elements_end_to_end_shape_and_values():
    job_rollup = _rollup_with_two_dependent_operations()
    enriched = calculate_delay_elements_for_jobs([job_rollup])[0]

    assert enriched["job_id"] == "JOB-TEST-1"
    ops = {op["operation_id"]: op for op in enriched["operations"]}
    op_a, op_b = ops["op-a"], ops["op-b"]

    # op-a: fully computable
    assert op_a["time_overrun_ratio"] == pytest.approx(350 / 480)
    assert op_a["predecessor_time_overrun_ratio"] is None  # independent
    assert op_a["operator_pace_ratio"] == pytest.approx(500 / 400)
    assert op_a["material_shortfall_ratio"] == pytest.approx(100 / 40)
    assert op_a["components"][0]["vendor"]["vendor_lead_time_ratio"] == pytest.approx(21 / 14)
    assert op_a["composite_risk_score"] is not None
    assert isinstance(op_a["is_delayed"], bool)

    # op-b: not started, but dependent - predecessor + operator-history fallback both kick in
    assert op_b["time_overrun_ratio"] is None
    assert op_b["predecessor_time_overrun_ratio"] == op_a["time_overrun_ratio"]
    assert op_b["predicted_overrun_hours"] is not None  # operator-history fallback, not None
    assert op_b["operator_pace_ratio"] == pytest.approx(500 / 400)


def test_calculate_delay_elements_does_not_mutate_input():
    job_rollup = _rollup_with_two_dependent_operations()
    import copy
    original = copy.deepcopy(job_rollup)
    calculate_delay_elements_for_jobs([job_rollup])
    assert job_rollup == original


def test_calculate_delay_elements_respects_custom_weights_and_threshold():
    job_rollup = _rollup_with_two_dependent_operations()
    # Weight EVERYTHING on time_overrun_ratio, and set a very low threshold
    # so a small ratio still counts as delayed - proves both params are wired through.
    custom_weights = {
        "time_overrun_ratio": 1.0,
        "operator_pace_ratio": 0.0,
        "material_shortfall_ratio": 0.0,
        "supplier_reliability": 0.0,
    }
    enriched = calculate_delay_elements_for_jobs(
        [job_rollup], risk_weights=custom_weights, delay_threshold=0.1,
    )[0]
    op_a = next(op for op in enriched["operations"] if op["operation_id"] == "op-a")
    assert op_a["composite_risk_score"] == pytest.approx(op_a["time_overrun_ratio"])
    assert op_a["is_delayed"] is True  # 0.729 > 0.1


def test_calculate_delay_elements_handles_empty_job():
    enriched = calculate_delay_elements_for_jobs([{"job_id": "EMPTY", "operations": []}])[0]
    assert enriched == {"job_id": "EMPTY", "operations": []}


# ─── multiple jobs: calculate_delay_elements_for_jobs ───────────────────────


def _shared_operator_history() -> list[dict]:
    """As build_job_rollups() would actually populate it: the SAME operator's
    last_10_work_orders already spans BOTH jobs (the tenant-wide operator
    index does that at the rollup stage, before elements.py ever sees it) -
    this is what a real shared operator looks like by the time it reaches
    calculate_delay_elements."""
    return [
        make_history_entry(elapsed=120, scheduled=100, completed_on="2026-09-05"),  # from job A
        make_history_entry(elapsed=200, scheduled=250, completed_on="2026-09-01"),  # from job B
    ]


def test_calculate_delay_elements_for_jobs_processes_every_job():
    shared_history = _shared_operator_history()
    shared_operator = make_operator(shared_history)

    job_a = {
        "job_id": "JOB-A",
        "operations": [
            make_op(
                operation_id="a1", depends_on_operation_ids=[],
                actual_duration_minutes=600, expected_duration_minutes=400,
                job_quantity=100.0, current_done_quantity=100.0,
                operators=[shared_operator],
                components=[make_component(required=100, available=100)],  # no shortage
            ),
        ],
    }
    job_b = {
        "job_id": "JOB-B",
        "operations": [
            make_op(
                operation_id="b1", depends_on_operation_ids=[],
                actual_duration_minutes=100, expected_duration_minutes=200,
                job_quantity=50.0, current_done_quantity=50.0,
                operators=[shared_operator],
                components=[make_component(required=100, available=20)],  # shortage: 5.0
            ),
            make_op(
                # depends on "a1" - a REAL operation id, but in the OTHER job.
                # Must NOT resolve - a1's time_overrun_ratio must never leak
                # from job A's map into job B's lookup.
                operation_id="b2", depends_on_operation_ids=["a1"],
                actual_duration_minutes=None, expected_duration_minutes=150,
                job_quantity=50.0, current_done_quantity=0.0,
                operators=[], components=[],
            ),
        ],
    }

    enriched = calculate_delay_elements_for_jobs([job_a, job_b])

    assert [j["job_id"] for j in enriched] == ["JOB-A", "JOB-B"]

    a1 = enriched[0]["operations"][0]
    b1, b2 = enriched[1]["operations"]

    # job A's own numbers, untouched by job B
    assert a1["time_overrun_ratio"] == pytest.approx(600 / 400)
    assert a1["material_shortfall_ratio"] == 0.0

    # job B's own numbers, untouched by job A - same operator, DIFFERENT
    # operations, so operator_pace_ratio (built from the operator's full
    # cross-job history) is identical, but everything operation-specific differs.
    assert b1["time_overrun_ratio"] == pytest.approx(100 / 200)
    assert b1["material_shortfall_ratio"] == pytest.approx(100 / 20)
    assert a1["operator_pace_ratio"] == b1["operator_pace_ratio"] == pytest.approx(
        (120 / 100 + 200 / 250) / 2
    )

    # b2 depends on "a1" - which exists ONLY in job A, not job B. Must resolve
    # to None (no known predecessor WITHIN THIS JOB), never reach across into
    # job A's results.
    assert b2["predecessor_time_overrun_ratio"] is None


def test_calculate_delay_elements_for_jobs_empty_list():
    assert calculate_delay_elements_for_jobs([]) == []


def test_calculate_delay_elements_for_jobs_result_independent_of_batching():
    """Processing [job_a, job_b] together must give the exact same per-job
    result as processing each one separately - proves jobs genuinely don't
    interact with each other regardless of how many are passed in at once."""
    job_a = {"job_id": "A", "operations": [make_op(operation_id="a1")]}
    job_b = {"job_id": "B", "operations": [make_op(operation_id="b1", actual_duration_minutes=None)]}

    together = calculate_delay_elements_for_jobs([job_a, job_b])
    separately = [
        calculate_delay_elements_for_jobs([job_a])[0],
        calculate_delay_elements_for_jobs([job_b])[0],
    ]
    assert together == separately
