"""Snapshots of real scoring events.

Built from the recorded Section 1 output (``review/fixtures/section1_job.json``)
and a real review of it, so these fail if the Risk Engine's shape changes
under the allow-list.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from maxxflow_features.lake import LakeIO
from m3_production_delay.llm_agents.review_agent import ReviewAgent
from m3_production_delay.review.pipeline import review_job
from m3_production_delay.snapshots.snapshot import (
    OPERATION_FIELDS,
    build_snapshot,
    write_snapshots,
)
from m3_production_delay.snapshots.store import LAYER, MODULE, SNAPSHOT_SCHEMA

_FIXTURE = (
    Path(__file__).resolve().parents[4]
    / "modules" / "m3_production_delay" / "review" / "fixtures" / "section1_job.json"
)
# The vector and cutoff the fixture was scored with (review/fixtures/build_fixture.py).
WEIGHTS = {
    "time_overrun_ratio": 0.40,
    "operator_pace_ratio": 0.30,
    "material_shortfall_ratio": 0.15,
    "supplier_reliability": 0.05,
}
THRESHOLD = 1.0
SCORED_AT = "2026-09-24T09:30:00+10:00"
LATER = "2026-09-24T13:45:00+10:00"


@pytest.fixture
def job() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def insight(job):
    reviewed = review_job(job, WEIGHTS, THRESHOLD, ReviewAgent())
    return dataclasses.replace(reviewed, generated_at=SCORED_AT)


@pytest.fixture
def lake(tmp_path) -> LakeIO:
    return LakeIO(SimpleNamespace(lake_uri=tmp_path.as_uri(), lake_storage_options={}))


def _build(job, insight):
    scored = dt.datetime.fromisoformat(SCORED_AT)
    return build_snapshot(
        job,
        insight,
        tenant="demo",
        risk_weights=WEIGHTS,
        threshold=THRESHOLD,
        scored_at=scored,
        written_at=scored,
    )


def _snapshot_files(tmp_path) -> list[Path]:
    return sorted((tmp_path / "demo" / MODULE / LAYER / "snapshots").rglob("*.json"))


# ---------------------------------------------------------------- content


def test_every_signal_is_kept_including_the_ones_that_did_not_fire(job, insight):
    record = _build(job, insight)
    assert record["schema"] == SNAPSHOT_SCHEMA
    assert len(record["operations"]) == len(job["operations"])
    for kept, source in zip(record["operations"], job["operations"]):
        for field in OPERATION_FIELDS:
            assert field in kept
        assert kept["time_overrun_ratio"] == source["time_overrun_ratio"]
        assert kept["operator_pace_ratio"] == source["operator_pace_ratio"]
        assert kept["composite_risk_score"] == source["composite_risk_score"]
        # Not attached to the operation by the engine; recomputed here.
        assert "supplier_reliability" in kept


def test_the_weights_threshold_and_full_insight_are_recorded(job, insight):
    record = _build(job, insight)
    assert record["risk_weights"] == WEIGHTS
    assert record["delay_threshold"] == THRESHOLD
    assert record["job_id"] == job["job_id"]
    assert record["insight"] == json.loads(json.dumps(insight.to_dict()))


def test_no_operator_identity_reaches_the_lake(job, insight):
    text = json.dumps(_build(job, insight))
    assert "operators" not in text
    assert "hmac-operator" not in text
    assert "last_10_work_orders" not in text
    assert "last_10_purchase_orders" not in text


def test_a_field_the_engine_adds_later_does_not_leak_in(job, insight):
    job["operations"][0]["some_new_engine_field"] = "unreviewed"
    assert "some_new_engine_field" not in json.dumps(_build(job, insight))


def test_an_infinite_shortfall_survives_as_a_readable_value(job, insight):
    zero_stock = copy.deepcopy(job)
    zero_stock["operations"][0]["material_shortfall_ratio"] = math.inf
    record = _build(zero_stock, insight)
    assert math.isinf(float(record["operations"][0]["material_shortfall_ratio"]))
    json.dumps(record, allow_nan=False)


# ---------------------------------------------------------------- writing


def test_one_snapshot_is_written_per_insight(job, insight, lake, tmp_path):
    report = write_snapshots([job], [insight], tenant="demo", risk_weights=WEIGHTS,
                             threshold=THRESHOLD, lake=lake)
    assert (report.written, report.failed) == (1, 0)
    files = _snapshot_files(tmp_path)
    assert len(files) == 1
    # 09:30 at +10:00 is 23:30 the previous day in UTC.
    assert files[0].parent.name == "dt=2026-09-23"
    assert json.loads(files[0].read_text(encoding="utf-8"))["job_id"] == job["job_id"]


def test_re_scoring_the_same_mo_adds_a_snapshot_and_keeps_the_first(
    job, insight, lake, tmp_path
):
    write_snapshots([job], [insight], tenant="demo", risk_weights=WEIGHTS,
                    threshold=THRESHOLD, lake=lake)
    first = _snapshot_files(tmp_path)[0].read_text(encoding="utf-8")

    rescored = dataclasses.replace(insight, generated_at=LATER)
    write_snapshots([job], [rescored], tenant="demo", risk_weights=WEIGHTS,
                    threshold=THRESHOLD, lake=lake)

    files = _snapshot_files(tmp_path)
    assert len(files) == 2
    assert first in [f.read_text(encoding="utf-8") for f in files]


def test_a_failing_lake_is_reported_never_raised(job, insight):
    class BrokenLake:
        def write_json(self, *args, **kwargs):
            raise OSError("storage unreachable")

    report = write_snapshots([job], [insight], tenant="demo", risk_weights=WEIGHTS,
                             threshold=THRESHOLD, lake=BrokenLake())
    assert (report.written, report.failed) == (0, 1)


def test_an_unreachable_lake_at_construction_is_reported_never_raised(
    job, insight, monkeypatch
):
    def explode():
        raise RuntimeError("bad LAKE_URI")

    monkeypatch.setattr("maxxflow_features.lake.get_lake", explode)
    report = write_snapshots([job], [insight], tenant="demo", risk_weights=WEIGHTS,
                             threshold=THRESHOLD)
    assert (report.written, report.failed) == (0, 1)


def test_nothing_to_snapshot_touches_nothing():
    class NeverCalled:
        def write_json(self, *args, **kwargs):
            raise AssertionError("no insight, no write")

    report = write_snapshots([], [], tenant="demo", risk_weights=WEIGHTS,
                             threshold=THRESHOLD, lake=NeverCalled())
    assert (report.written, report.failed) == (0, 0)


def test_an_insight_without_its_own_stamp_uses_the_clock(job, insight, lake, tmp_path):
    unstamped = dataclasses.replace(insight, generated_at=None)
    report = write_snapshots([job], [unstamped], tenant="demo", risk_weights=WEIGHTS,
                             threshold=THRESHOLD, lake=lake)
    assert report.written == 1
    record = json.loads(_snapshot_files(tmp_path)[0].read_text(encoding="utf-8"))
    assert record["scored_at"]


# ---------------------------------------------------------------- pipeline hook


@pytest.fixture
def run_harness(job, insight, monkeypatch):
    """review.pipeline.run with every IO seam replaced, recording what it calls."""
    calls = {"published": [], "snapshotted": []}

    class FakeOrchestrator:
        def resolve_risk_weights(self, request):
            return dict(WEIGHTS)

        def review_jobs(self, scored, *, weights, threshold):
            return [insight]

    monkeypatch.setattr(
        "m3_production_delay.rule_engine.dal.read_delay_tables", lambda tenant: ({}, None)
    )
    monkeypatch.setattr(
        "m3_production_delay.rule_engine.rollup.build_job_rollups",
        lambda tables, md, job_references=None: [job],
    )
    monkeypatch.setattr(
        "m3_production_delay.rule_engine.elements.calculate_delay_elements_for_jobs",
        lambda rollups, risk_weights, delay_threshold: [job],
    )
    monkeypatch.setattr(
        "m3_production_delay.orchestrator.ProductionDelayOrchestrator", FakeOrchestrator
    )
    monkeypatch.setattr(
        "m3_production_delay.review.pipeline.publish_insights",
        lambda insights, tenant, dry_run: calls["published"].append(dry_run),
    )
    monkeypatch.setattr(
        "m3_production_delay.snapshots.write_snapshots",
        lambda scored, insights, **kwargs: calls["snapshotted"].append((scored, insights, kwargs)),
    )
    return calls


def test_a_review_run_snapshots_what_it_published(run_harness, job, insight):
    from m3_production_delay.review.pipeline import run

    run(tenant="demo", threshold=THRESHOLD)
    assert run_harness["published"] == [False]
    [(scored, insights, kwargs)] = run_harness["snapshotted"]
    assert scored == [job]
    assert insights == [insight]
    assert kwargs == {"tenant": "demo", "risk_weights": WEIGHTS, "threshold": THRESHOLD}


def test_a_dry_run_writes_no_snapshot(run_harness):
    from m3_production_delay.review.pipeline import run

    run(tenant="demo", threshold=THRESHOLD, dry_run=True)
    assert run_harness["published"] == [True]
    assert run_harness["snapshotted"] == []
