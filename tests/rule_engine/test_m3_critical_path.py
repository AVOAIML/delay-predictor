from __future__ import annotations

import pytest

from m3_production_delay.rule_engine.critical_path import (
    analyze_critical_path,
    apply_cascade_risk,
    cascading_overruns,
)


def _op(operation_id: str, duration: float, *predecessors: str) -> dict:
    return {
        "operation_id": operation_id,
        "expected_duration_minutes": duration,
        "depends_on_operation_ids": list(predecessors),
    }


def test_cpm_selects_longest_dependency_path_not_every_dependency():
    analysis = analyze_critical_path([
        _op("critical-a", 100),
        _op("critical-b", 50, "critical-a"),
        _op("short-a", 20),
        _op("short-b", 20, "short-a"),
        _op("independent", 10),
    ])

    assert analysis.valid
    assert analysis.project_duration_minutes == 150
    assert analysis.nodes["critical-a"].is_critical
    assert analysis.nodes["critical-b"].critical_predecessor_ids == ("critical-a",)
    assert not analysis.nodes["short-a"].is_critical
    assert analysis.nodes["short-b"].total_float_minutes == 110
    assert not analysis.nodes["independent"].is_critical


def test_only_overrun_on_critical_edge_cascades():
    analysis = analyze_critical_path([
        _op("critical-a", 100),
        _op("critical-b", 50, "critical-a"),
        _op("short-a", 20),
        _op("short-b", 20, "short-a"),
    ])
    signals = cascading_overruns(
        analysis,
        {"critical-a": 1.30, "critical-b": None, "short-a": 2.0, "short-b": None},
    )

    assert signals["critical-b"].ratio == pytest.approx(1.30)
    assert signals["critical-b"].source_operation_ids == ("critical-a",)
    assert signals["short-b"] is None


def test_cascade_propagates_through_successive_critical_operations():
    analysis = analyze_critical_path([
        _op("a", 100), _op("b", 50, "a"), _op("c", 25, "b"),
    ])
    signals = cascading_overruns(analysis, {"a": 1.35, "b": None, "c": None})

    assert signals["b"].ratio == pytest.approx(1.35)
    assert signals["c"].ratio == pytest.approx(1.35)
    assert signals["c"].source_operation_ids == ("a",)


def test_cycle_disables_cascade_instead_of_crashing_job_scoring():
    analysis = analyze_critical_path([_op("a", 10, "b"), _op("b", 10, "a")])

    assert not analysis.valid
    assert "cycle" in analysis.error
    assert cascading_overruns(analysis, {"a": 2.0, "b": 2.0}) == {}


def test_cascade_overlay_always_increases_existing_score_and_bootstraps_missing_score():
    assert apply_cascade_risk(0.8, 1.3) == pytest.approx(1.3)
    assert apply_cascade_risk(1.8, 1.3) == pytest.approx(2.1)
    assert apply_cascade_risk(None, 1.3) == pytest.approx(1.3)
    assert apply_cascade_risk(0.8, None) == pytest.approx(0.8)
