"""The outcome sweep: labels, the bookmark, and what it refuses to do.

The database is an in-memory stand-in that honours the sweep's ``since``
filter the way the real query does, so the watermark behaviour is exercised
rather than assumed. The lake is a real ``file://`` LakeIO.
"""

from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from maxxflow_data.masterdata_map import MasterDataMap
from maxxflow_features.lake import LakeIO
from m3_production_delay.snapshots.outcomes import (
    DEFAULT_LOOKBACK,
    LABEL_VERSION,
    label_outcome,
    sweep_outcomes,
)
from m3_production_delay.snapshots.store import (
    LAYER,
    MODULE,
    OUTCOME_SCHEMA,
    WATERMARK_NAME,
    outcome_name,
)

T = dt.datetime  # database-native timestamps are naive

DONE, CANCELLED, WO_DONE = "st-mo-done", "st-mo-cancelled", "st-wo-done"
MD = MasterDataMap(
    code_to_id={
        ("MO_STATUS", "DONE"): DONE,
        ("MO_STATUS", "CANCELLED"): CANCELLED,
        ("WORK_ORDER_STATUS", "DONE"): WO_DONE,
    }
)


def _wo(mo_id, *, scheduled_end, actual_end, expected=180, real=200):
    return {
        "mo_id": mo_id, "id": f"wo-{mo_id}-{scheduled_end:%H%M}", "operation_id": "op-1",
        "expected_duration": expected, "real_duration": real,
        "scheduled_start": scheduled_end - dt.timedelta(minutes=expected),
        "scheduled_end": scheduled_end, "actual_start": None, "actual_end": actual_end,
        "status_id": WO_DONE,
    }


class FakeDB:
    """Answers the two sweep queries from in-memory rows, applying ``since``."""

    def __init__(self, mos, work_orders):
        self.mos = mos
        self.work_orders = work_orders
        self.since_seen = []

    def query(self, sql, params=None, *, tenant=None):
        params = params or {}
        if "FROM manufacturing_orders" in sql:
            since = params.get("since")
            self.since_seen.append(since)
            rows = [m for m in self.mos if since is None or m["completed_at"] > since]
            return pd.DataFrame(sorted(rows, key=lambda m: m["completed_at"]))
        if "FROM work_orders" in sql:
            wanted = set(params["mo_ids"])
            return pd.DataFrame([w for w in self.work_orders if w["mo_id"] in wanted])
        raise AssertionError(f"unexpected query: {sql}")


@pytest.fixture(autouse=True)
def _master_data(monkeypatch):
    monkeypatch.setattr(
        "m3_production_delay.snapshots.outcomes.load_md_map", lambda da, tenant: MD
    )


@pytest.fixture
def lake(tmp_path) -> LakeIO:
    return LakeIO(SimpleNamespace(lake_uri=tmp_path.as_uri(), lake_storage_options={}))


def _mo(mo_id, reference, completed_at, status=DONE):
    return {
        "id": mo_id, "reference": reference, "scheduled_date": T(2026, 9, 20, 8, 0),
        "confirmed_at": T(2026, 9, 20, 7, 0), "completed_at": completed_at, "status_id": status,
    }


@pytest.fixture
def db() -> FakeDB:
    return FakeDB(
        mos=[
            _mo("mo-1", "WH/MO/00142", T(2026, 9, 23, 21, 15)),  # 1h15m late
            _mo("mo-2", "WH/MO/00151", T(2026, 9, 23, 23, 40)),  # early
        ],
        work_orders=[
            _wo("mo-1", scheduled_end=T(2026, 9, 23, 18, 0), actual_end=T(2026, 9, 23, 19, 0)),
            _wo("mo-1", scheduled_end=T(2026, 9, 23, 20, 0), actual_end=T(2026, 9, 23, 21, 15)),
            _wo("mo-2", scheduled_end=T(2026, 9, 25, 12, 0), actual_end=T(2026, 9, 23, 23, 40),
                real=150),
        ],
    )


def _outcome(lake, reference):
    return lake.read_json(LAYER, outcome_name(reference), tenant="demo", module=MODULE)


def _watermark(lake):
    return lake.read_json(LAYER, WATERMARK_NAME, tenant="demo", module=MODULE)


# ---------------------------------------------------------------- the label


def test_finishing_after_the_latest_planned_end_is_late():
    label = label_outcome(
        T(2026, 9, 23, 21, 15),
        [{"scheduled_end": T(2026, 9, 23, 18, 0)}, {"scheduled_end": T(2026, 9, 23, 20, 0)}],
    )
    assert label["planned_finish"] == T(2026, 9, 23, 20, 0)  # the LATEST end, not the first
    assert label["was_late"] is True
    assert label["hours_late"] == pytest.approx(1.25)
    assert label["version"] == LABEL_VERSION


def test_finishing_before_the_planned_end_is_early_with_negative_hours():
    label = label_outcome(T(2026, 9, 23, 18, 0), [{"scheduled_end": T(2026, 9, 23, 20, 0)}])
    assert label["was_late"] is False
    assert label["hours_late"] == pytest.approx(-2.0)


def test_no_planned_end_means_no_label_rather_than_a_guess():
    label = label_outcome(T(2026, 9, 23, 18, 0), [{"scheduled_end": None}])
    assert label["was_late"] is None
    assert label["hours_late"] is None


def test_the_duration_ratio_needs_every_actual_duration():
    complete = label_outcome(None, [{"expected_duration": 100, "real_duration": 150},
                                     {"expected_duration": 100, "real_duration": 50}])
    assert complete["duration_overrun_ratio"] == pytest.approx(1.0)
    partial = label_outcome(None, [{"expected_duration": 100, "real_duration": 150},
                                    {"expected_duration": 100, "real_duration": None}])
    assert partial["duration_overrun_ratio"] is None
    assert partial["actual_minutes_total"] is None


# ---------------------------------------------------------------- the sweep


def test_the_first_sweep_records_all_history_and_sets_the_bookmark(db, lake):
    report = sweep_outcomes("demo", lake=lake, data_access=db)

    assert db.since_seen == [None]
    assert (report.found, report.written, report.failed) == (2, 2, 0)
    assert _outcome(lake, "WH/MO/00142")["label"]["was_late"] is True
    assert _outcome(lake, "WH/MO/00151")["label"]["was_late"] is False
    assert _watermark(lake)["completed_at"] == T(2026, 9, 23, 23, 40).isoformat()
    assert report.watermark_after == T(2026, 9, 23, 23, 40).isoformat()


def test_the_next_sweep_starts_before_the_bookmark_by_the_lookback(db, lake):
    sweep_outcomes("demo", lake=lake, data_access=db)
    db.mos.append(_mo("mo-3", "WH/MO/00163", T(2026, 9, 24, 3, 0)))
    db.work_orders.append(
        _wo("mo-3", scheduled_end=T(2026, 9, 24, 1, 0), actual_end=T(2026, 9, 24, 3, 0))
    )

    report = sweep_outcomes("demo", lake=lake, data_access=db)

    assert db.since_seen[-1] == T(2026, 9, 23, 23, 40) - DEFAULT_LOOKBACK
    assert _outcome(lake, "WH/MO/00163")["label"]["was_late"] is True
    assert _watermark(lake)["completed_at"] == T(2026, 9, 24, 3, 0).isoformat()
    assert report.watermark_before == T(2026, 9, 23, 23, 40).isoformat()


def test_re_sweeping_the_overlap_rewrites_the_same_outcome(db, lake):
    sweep_outcomes("demo", lake=lake, data_access=db)
    first = _outcome(lake, "WH/MO/00142")
    sweep_outcomes("demo", lake=lake, data_access=db)  # lookback re-finds both MOs
    again = _outcome(lake, "WH/MO/00142")
    first.pop("swept_at"), again.pop("swept_at")
    assert first == again


def test_a_cancelled_mo_is_skipped_not_labelled(db, lake):
    db.mos.append(_mo("mo-9", "WH/MO/00199", T(2026, 9, 23, 22, 0), status=CANCELLED))
    report = sweep_outcomes("demo", lake=lake, data_access=db)
    assert report.skipped == 1
    assert _outcome(lake, "WH/MO/00199") is None


def test_one_failed_write_holds_the_bookmark_so_the_window_is_retried(db, lake):
    class FailsOneOutcome(LakeIO):
        def write_json(self, obj, layer, name, *, tenant, module):
            if name == outcome_name("WH/MO/00151"):
                raise OSError("storage blip")
            return super().write_json(obj, layer, name, tenant=tenant, module=module)

    flaky = FailsOneOutcome(lake.settings)
    report = sweep_outcomes("demo", lake=flaky, data_access=db)

    assert (report.written, report.failed) == (1, 1)
    assert _watermark(lake) is None
    assert report.watermark_after is None


def test_a_dry_run_writes_nothing_and_leaves_the_bookmark(db, lake):
    report = sweep_outcomes("demo", lake=lake, data_access=db, dry_run=True)
    assert report.found == 2
    assert report.written == 0
    assert _outcome(lake, "WH/MO/00142") is None
    assert _watermark(lake) is None


def test_nothing_completed_is_a_quiet_no_op(lake):
    report = sweep_outcomes("demo", lake=lake, data_access=FakeDB(mos=[], work_orders=[]))
    assert (report.found, report.written) == (0, 0)
    assert _watermark(lake) is None


# ---------------------------------------------------------------- the record


def test_an_outcome_keeps_the_raw_timings_behind_its_label(db, lake):
    sweep_outcomes("demo", lake=lake, data_access=db)
    record = _outcome(lake, "WH/MO/00142")

    assert record["schema"] == OUTCOME_SCHEMA
    assert record["job_id"] == "WH/MO/00142"  # exact reference, not the file key
    assert record["status"] == "DONE"
    assert record["scheduled_date"]  # recorded even though the label ignores it
    assert len(record["work_orders"]) == 2
    first, second = record["work_orders"]
    assert first["status"] == "DONE"
    assert first["finished_late"] is True
    assert first["duration_overrun_ratio"] == pytest.approx(200 / 180)
    assert set(first) >= {"scheduled_end", "actual_end", "expected_duration_minutes"}


def test_outcome_timestamps_carry_a_timezone(db, lake):
    sweep_outcomes("demo", lake=lake, data_access=db)
    record = _outcome(lake, "WH/MO/00142")
    for value in (record["completed_at"], record["label"]["planned_finish"],
                  record["work_orders"][0]["scheduled_end"]):
        assert dt.datetime.fromisoformat(value).tzinfo is not None


def test_no_operator_identity_is_read_or_written(db, lake):
    sweep_outcomes("demo", lake=lake, data_access=db)
    text = json.dumps(_outcome(lake, "WH/MO/00142"))
    assert "operator" not in text.replace("operation_id", "")
