import json
import uuid

import fsspec

from maxxflow_core.settings import Settings
from m3_production_delay.snapshot_digest import (
    LakeSnapshotSource,
    LocalDirectorySnapshotSource,
    build_snapshot_digest,
    build_tenant_snapshot_digest,
    load_snapshots,
    snapshot_prefix,
)


def _snapshot(
    *,
    tenant="demo",
    scored_at="2026-09-24T01:00:00+10:00",
    is_delayed=True,
    status="fallback_template",
    judge_approved=False,
    operations,
):
    return {
        "schema": "m3.snapshot.v1",
        "tenant": tenant,
        "scored_at": scored_at,
        "operations": operations,
        "insight": {
            "is_delayed": is_delayed,
            "status": status,
            "judge": {"approved": judge_approved},
        },
    }


def _operation(**ratios):
    return dict(ratios)


# --- load_snapshots -----------------------------------------------------


def test_load_snapshots_returns_empty_list_for_missing_directory(tmp_path):
    assert load_snapshots(tmp_path / "does_not_exist") == []


def test_load_snapshots_skips_malformed_and_wrong_schema_files(tmp_path):
    (tmp_path / "good.json").write_text(json.dumps(_snapshot(operations=[_operation()])))
    (tmp_path / "not_json.json").write_text("{not valid json")
    (tmp_path / "wrong_schema.json").write_text(json.dumps({"schema": "other.v1"}))
    (tmp_path / "not_an_object.json").write_text(json.dumps(["a", "list"]))

    snapshots = load_snapshots(tmp_path)

    assert len(snapshots) == 1
    assert snapshots[0]["schema"] == "m3.snapshot.v1"


# --- build_snapshot_digest -----------------------------------------------


def test_returns_none_when_no_snapshot_matches_tenant():
    snapshots = [_snapshot(tenant="other", operations=[_operation()])]
    assert build_snapshot_digest(snapshots, "demo") is None


def test_counts_and_rates_are_computed_correctly():
    snapshots = [
        _snapshot(
            scored_at="2026-09-24T01:00:00+10:00",
            is_delayed=True,
            status="fallback_template",
            judge_approved=False,
            operations=[
                _operation(time_overrun_ratio=2.0, operator_pace_ratio=1.25),
                _operation(time_overrun_ratio=0.5, operator_pace_ratio=0.9),
            ],
        ),
        _snapshot(
            scored_at="2026-09-24T11:00:00+10:00",
            is_delayed=False,
            status="approved",
            judge_approved=True,
            operations=[_operation(time_overrun_ratio=1.5, operator_pace_ratio=1.3)],
        ),
    ]

    digest = build_snapshot_digest(snapshots, "demo")

    assert "n_scored_jobs=2" in digest
    assert "predicted_delayed_rate=1/2" in digest
    assert "judge_approved_rate=1/2" in digest
    # time_overrun_ratio fires above 1.0: two of the three operation values
    # (2.0, 1.5) clear it, 0.5 does not.
    assert "time_overrun: fired 2/3 operations (fires above 1.0), average value 1.33" in digest
    # operator_pace_ratio fires above 1.2: 1.25 and 1.3 clear it, 0.9 does not.
    assert "operator_skill: fired 2/3 operations (fires above 1.2), average value 1.15" in digest


def test_signals_with_no_recorded_values_are_omitted_not_zero_filled():
    snapshots = [_snapshot(operations=[_operation(time_overrun_ratio=2.0)])]
    digest = build_snapshot_digest(snapshots, "demo")
    assert "time_overrun:" in digest
    assert "operator_skill:" not in digest
    assert "material_availability:" not in digest
    assert "supplier_reliability:" not in digest


def test_non_numeric_and_boolean_signal_values_are_ignored():
    snapshots = [
        _snapshot(
            operations=[
                _operation(time_overrun_ratio=None),
                _operation(time_overrun_ratio=True),  # bool is not a real ratio
                _operation(time_overrun_ratio=2.0),
            ]
        )
    ]
    digest = build_snapshot_digest(snapshots, "demo")
    assert "time_overrun: fired 1/1 operations" in digest


def test_max_snapshots_keeps_the_most_recent_by_scored_at():
    old = _snapshot(scored_at="2026-01-01T00:00:00+10:00", operations=[_operation(time_overrun_ratio=5.0)])
    recent = _snapshot(scored_at="2026-09-24T00:00:00+10:00", operations=[_operation(time_overrun_ratio=1.5)])
    digest = build_snapshot_digest([old, recent], "demo", max_snapshots=1)
    assert "n_scored_jobs=1" in digest
    assert "average value 1.50" in digest


# --- build_tenant_snapshot_digest (load + build) --------------------------


def test_build_tenant_snapshot_digest_reads_from_directory(tmp_path):
    (tmp_path / "snap1.json").write_text(
        json.dumps(_snapshot(operations=[_operation(supplier_reliability=1.5)]))
    )
    digest = build_tenant_snapshot_digest("demo", directory=tmp_path)
    assert digest is not None
    assert "n_scored_jobs=1" in digest
    assert "supplier_reliability: fired 1/1 operations (fires above 1.0)" in digest


def test_build_tenant_snapshot_digest_returns_none_for_empty_directory(tmp_path):
    assert build_tenant_snapshot_digest("demo", directory=tmp_path) is None


# --- LakeSnapshotSource (MinIO/Azure, via fsspec) -------------------------
#
# `memory://` is fsspec's own in-process filesystem — it exercises the real
# glob/open code path in LakeSnapshotSource without a live MinIO or Azure
# endpoint. LakeSnapshotSource has no scheme-specific branching (that's the
# point: fsspec.core.url_to_fs dispatches on LAKE_URI's scheme, identically
# for s3://, abfss:// and memory://), so this proves the same code that will
# run against MinIO/Azure in production.


def _memory_lake_settings() -> Settings:
    # A fresh, unique root per test — MemoryFileSystem is a process-wide
    # singleton, so a shared prefix would leak state between tests.
    #
    # Settings fields are declared with `alias="LAKE_URI"` and no
    # `populate_by_name`, so only the alias (not the Python attribute name
    # `lake_uri`) actually binds a constructor kwarg — passing `lake_uri=`
    # here would silently be dropped (`extra="ignore"`) and fall through to
    # whatever config/profiles/local.env sets instead.
    return Settings(LAKE_URI=f"memory://lake-{uuid.uuid4().hex}")


def _write_snapshot(settings: Settings, tenant_id: str, name: str, payload) -> None:
    prefix = snapshot_prefix(tenant_id, settings=settings)
    fs, root = fsspec.core.url_to_fs(prefix)
    fs.makedirs(root, exist_ok=True)
    body = payload if isinstance(payload, (str, bytes)) else json.dumps(payload)
    fs.pipe_file(f"{root}/{name}.json", body.encode() if isinstance(body, str) else body)


def test_snapshot_prefix_matches_the_lake_io_tenant_module_layout():
    settings = Settings(LAKE_URI="s3://maxxflow-lake")
    assert snapshot_prefix("demo", settings=settings) == (
        "s3://maxxflow-lake/demo/m3_production_delay/snapshots"
    )


def test_lake_snapshot_source_reads_only_the_requested_tenants_objects():
    settings = _memory_lake_settings()
    _write_snapshot(
        settings, "tenant-a", "snap1", _snapshot(tenant="tenant-a", operations=[_operation()])
    )
    _write_snapshot(
        settings, "tenant-b", "snap1", _snapshot(tenant="tenant-b", operations=[_operation()])
    )

    tenant_a_snapshots = LakeSnapshotSource(settings).list_snapshots("tenant-a")

    assert len(tenant_a_snapshots) == 1
    assert tenant_a_snapshots[0]["tenant"] == "tenant-a"


def test_lake_snapshot_source_skips_malformed_json_and_wrong_schema():
    settings = _memory_lake_settings()
    _write_snapshot(settings, "demo", "good", _snapshot(operations=[_operation(time_overrun_ratio=2.0)]))
    _write_snapshot(settings, "demo", "bad_json", "{not valid json")
    _write_snapshot(settings, "demo", "wrong_schema", {"schema": "other.v1"})

    snapshots = LakeSnapshotSource(settings).list_snapshots("demo")

    assert len(snapshots) == 1
    assert snapshots[0]["schema"] == "m3.snapshot.v1"


def test_lake_snapshot_source_returns_empty_list_when_nothing_is_stored():
    settings = _memory_lake_settings()
    assert LakeSnapshotSource(settings).list_snapshots("demo") == []


def test_lake_snapshot_source_storage_failure_returns_empty_list_not_raise():
    # An unsupported/malformed scheme makes fsspec.core.url_to_fs raise —
    # standing in for a real backend being unreachable/misconfigured.
    settings = Settings(LAKE_URI="not-a-real-scheme://wherever")
    assert LakeSnapshotSource(settings).list_snapshots("demo") == []


def test_build_tenant_snapshot_digest_defaults_to_the_lake_and_isolates_tenants():
    settings = _memory_lake_settings()
    _write_snapshot(
        settings,
        "tenant-a",
        "snap1",
        _snapshot(tenant="tenant-a", operations=[_operation(time_overrun_ratio=2.0)]),
    )
    _write_snapshot(
        settings,
        "tenant-b",
        "snap1",
        _snapshot(tenant="tenant-b", operations=[_operation(time_overrun_ratio=99.0)]),
    )

    digest = build_tenant_snapshot_digest("tenant-a", settings=settings)

    assert digest is not None
    assert "n_scored_jobs=1" in digest
    assert "average value 2.00" in digest
    assert "99.0" not in digest


def test_build_tenant_snapshot_digest_returns_none_when_lake_is_empty():
    settings = _memory_lake_settings()
    assert build_tenant_snapshot_digest("demo", settings=settings) is None


def test_build_tenant_snapshot_digest_survives_a_raising_snapshot_source():
    class ExplodingSource:
        def list_snapshots(self, tenant_id: str):
            raise ConnectionError("simulated storage outage")

    assert build_tenant_snapshot_digest("demo", snapshot_source=ExplodingSource()) is None


def test_directory_override_takes_precedence_over_the_lake_default(tmp_path):
    (tmp_path / "snap.json").write_text(
        json.dumps(_snapshot(tenant="demo", operations=[_operation(time_overrun_ratio=2.0)]))
    )
    # No lake settings passed at all — proves `directory` alone is enough to
    # route to LocalDirectorySnapshotSource without ever touching the lake.
    digest = build_tenant_snapshot_digest("demo", directory=tmp_path)
    assert digest is not None
    assert "n_scored_jobs=1" in digest


def test_explicit_snapshot_source_overrides_both_lake_and_directory_defaults(tmp_path):
    source = LocalDirectorySnapshotSource(tmp_path)
    assert build_tenant_snapshot_digest("demo", snapshot_source=source) is None
