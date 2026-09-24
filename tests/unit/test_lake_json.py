"""LakeIO's JSON records: the bronze tier's one-object-per-event store.

Run against a real ``file://`` lake in a temp directory — the same fsspec code
path MinIO and ADLS take, minus the network.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from maxxflow_core.ports import LakeIO as LakeIOPort
from maxxflow_features.lake import LakeIO


@pytest.fixture
def lake(tmp_path) -> LakeIO:
    return LakeIO(SimpleNamespace(lake_uri=tmp_path.as_uri(), lake_storage_options={}))


def test_the_adapter_still_satisfies_the_port(lake):
    assert isinstance(lake, LakeIOPort)


def test_a_record_round_trips(lake):
    record = {"job_id": "WH/MO/00142", "values": [1, 2.5, None, "inf"], "nested": {"a": True}}
    uri = lake.write_json(
        record, "bronze", "snapshots/dt=2026-09-24/rec", tenant="demo", module="m3"
    )
    assert uri.endswith("/demo/m3/bronze/snapshots/dt=2026-09-24/rec.json")
    assert (
        lake.read_json("bronze", "snapshots/dt=2026-09-24/rec", tenant="demo", module="m3")
        == record
    )


def test_reading_a_missing_record_is_none(lake):
    assert lake.read_json("bronze", "outcomes/nope", tenant="demo", module="m3") is None


def test_a_non_finite_float_is_refused_and_nothing_is_written(lake):
    with pytest.raises(ValueError):
        lake.write_json({"x": float("inf")}, "bronze", "bad", tenant="demo", module="m3")
    assert lake.read_json("bronze", "bad", tenant="demo", module="m3") is None


def test_writing_again_replaces_the_record(lake):
    lake.write_json({"v": 1}, "bronze", "r", tenant="demo", module="m3")
    lake.write_json({"v": 2}, "bronze", "r", tenant="demo", module="m3")
    assert lake.read_json("bronze", "r", tenant="demo", module="m3") == {"v": 2}


def test_tenants_never_share_a_record(lake):
    lake.write_json({"owner": "acme"}, "bronze", "r", tenant="acme", module="m3")
    lake.write_json({"owner": "demo"}, "bronze", "r", tenant="demo", module="m3")
    assert lake.read_json("bronze", "r", tenant="acme", module="m3") == {"owner": "acme"}
    assert lake.read_json("bronze", "r", tenant="demo", module="m3") == {"owner": "demo"}


def test_parquet_paths_are_unchanged(lake):
    # M1/M2 write gold features through the same _uri; the new ext parameter
    # must not move them.
    assert lake._uri("gold", "features", tenant="t", module="m").endswith(
        "/t/m/gold/features.parquet"
    )
