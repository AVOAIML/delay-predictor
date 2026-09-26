"""Turns recorded ``m3.snapshot.v1`` records into a short, deterministic text
digest handed to the Weight Agent's LLM adjustment stage as extra evidence
(``WeightAgentRequest.history_digest``).

Counts and averages only — no fitting, no coefficients, no learned weight.
This module never proposes an adjustment or a weight; it only reports what a
tenant's recorded snapshots show, in the same "descriptive facts, not a
formula" spirit as the tenant profile the LLM already reasons over. The
resolver's fitted-weight route (``providers.FittedWeightsProvider``) remains
the only place a statistically defended weight can come from — this is a
deliberately weaker, LLM-facing substitute that needs no sample-size floor.

PRODUCTION SOURCE, AND A GAP THIS MODULE DOES NOT CLOSE: no code anywhere in
this repository currently *writes* an ``m3.snapshot.v1`` object to the lake —
the only place this schema exists today is a hand-captured local fixture
(``meta_data/snap_samples/``, untracked). ``LakeSnapshotSource`` below reads
from ``{LAKE_URI}/{tenant}/m3_production_delay/snapshots/*.json`` — the same
``{tenant}/{module}/...`` layout ``maxxflow_features.lake.LakeIO`` and
``maxxflow_mlops.model_artifact_store.ModelArtifactStore`` already use for
their own lake objects — but this is a *proposed* convention, not one
discovered in an existing writer. If a writer is ever added, it must target
this exact prefix (:func:`snapshot_prefix`), or the two will silently drift.
Until then, this source will simply find nothing and the digest stays
``None`` — the same safe "no history yet" behaviour as an empty fixture
directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import fsspec

from maxxflow_core.settings import Settings, get_settings
from m3_production_delay.review.evidence import FIRE_BASELINES
from m3_production_delay.rule_engine.elements import WEIGHT_AGENT_SIGNAL_TO_RISK_KEY

log = logging.getLogger(__name__)

SNAPSHOT_SCHEMA = "m3.snapshot.v1"

# The `{module}` path segment — matches this package's own name, same as
# ModelArtifactStore.uri()'s `{module}` (parsed from the MLflow model name).
SNAPSHOT_MODULE = "m3_production_delay"

# Local/demo fixture only — never read by default (see module docstring and
# build_tenant_snapshot_digest's `directory` parameter). Production reads
# through LakeSnapshotSource.
DEFAULT_SNAPSHOT_DIR = Path(__file__).parent / "meta_data" / "snap_samples"

# Reverse of the rule engine's own adapter map, so the digest speaks the
# Weight Agent's vocabulary (the prompt it's injected into does too) rather
# than the rule engine's. Only the four signals that map both ways — never
# `predecessor_time_overrun_ratio` / `critical_path_cascade_ratio`, which
# review/README.md's Fire baselines table already marks context-only, weight
# pinned to 0, and never a signal the rule engine renormalises over but the
# Weight Agent doesn't name (there are none: this map is a strict subset of
# FIRE_BASELINES's keys).
_RISK_KEY_TO_WEIGHT_AGENT_SIGNAL: dict[str, str] = {
    risk_key: signal for signal, risk_key in WEIGHT_AGENT_SIGNAL_TO_RISK_KEY.items()
}
_DIGEST_RISK_KEYS: tuple[str, ...] = tuple(
    risk_key for risk_key in _RISK_KEY_TO_WEIGHT_AGENT_SIGNAL if risk_key in FIRE_BASELINES
)


def load_snapshots(directory: Path) -> list[dict[str, Any]]:
    """Best-effort load of every ``*.json`` file in ``directory`` as an
    ``m3.snapshot.v1`` record.

    A file that fails to parse, or doesn't carry that schema tag, is skipped
    and logged, never raised — this is optional evidence for a prompt, not a
    required input anything downstream depends on existing.
    """
    snapshots: list[dict[str, Any]] = []
    if not directory.is_dir():
        return snapshots
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("snapshot_digest: skipping unreadable %s: %s", path, exc)
            continue
        if not isinstance(record, dict) or record.get("schema") != SNAPSHOT_SCHEMA:
            log.warning("snapshot_digest: skipping %s: not a %s record", path, SNAPSHOT_SCHEMA)
            continue
        snapshots.append(record)
    return snapshots


def build_snapshot_digest(
    snapshots: list[dict[str, Any]], tenant_id: str, *, max_snapshots: int = 50
) -> str | None:
    """Plain-text, deterministic summary of ``tenant_id``'s recorded
    snapshots: how often each signal fired, its average value, the judge's
    approval rate, and the predicted-delayed rate.

    ``None`` when there is nothing recorded for this tenant, so a caller can
    fall back to the prompt's own "no history available" default rather than
    inject an empty section.
    """
    tenant_snapshots = [s for s in snapshots if s.get("tenant") == tenant_id]
    # Most-recent-first is irrelevant to the counts below (every snapshot in
    # range is weighted equally) — sorting is only to make `max_snapshots`
    # keep the most recent ones, and to keep output order reproducible.
    tenant_snapshots.sort(key=lambda s: s.get("scored_at") or "")
    tenant_snapshots = tenant_snapshots[-max_snapshots:]
    if not tenant_snapshots:
        return None

    operations: list[dict[str, Any]] = []
    delayed_jobs = 0
    approved_jobs = 0
    for snapshot in tenant_snapshots:
        insight = snapshot.get("insight") or {}
        if insight.get("is_delayed"):
            delayed_jobs += 1
        judge = insight.get("judge") or {}
        if judge.get("approved") or insight.get("status") in (
            "approved",
            "approved_with_warnings",
        ):
            approved_jobs += 1
        operations.extend(snapshot.get("operations") or [])

    lines = [
        f"n_scored_jobs={len(tenant_snapshots)}",
        f"predicted_delayed_rate={delayed_jobs}/{len(tenant_snapshots)}",
        f"judge_approved_rate={approved_jobs}/{len(tenant_snapshots)}",
    ]
    for risk_key in _DIGEST_RISK_KEYS:
        values = [
            op[risk_key]
            for op in operations
            if isinstance(op.get(risk_key), (int, float)) and not isinstance(op.get(risk_key), bool)
        ]
        if not values:
            continue
        baseline = FIRE_BASELINES[risk_key]
        fired = sum(1 for value in values if value > baseline)
        average = sum(values) / len(values)
        signal_name = _RISK_KEY_TO_WEIGHT_AGENT_SIGNAL[risk_key]
        lines.append(
            f"{signal_name}: fired {fired}/{len(values)} operations "
            f"(fires above {baseline}), average value {average:.2f}"
        )
    return "\n".join(lines)


@runtime_checkable
class SnapshotSource(Protocol):
    """Where :func:`build_tenant_snapshot_digest` gets a tenant's raw
    ``m3.snapshot.v1`` records from. :class:`LakeSnapshotSource` (production)
    and :class:`LocalDirectorySnapshotSource` (test/demo fixture) are the two
    shipped implementations — a test can supply any object with this one
    method instead of touching real storage, same pattern as
    ``llm_agents.weight_agent.providers.FittedWeightsProvider``.
    """

    def list_snapshots(self, tenant_id: str) -> list[dict[str, Any]]: ...


def snapshot_prefix(tenant_id: str, *, settings: Settings | None = None) -> str:
    """The lake URI ``tenant_id``'s snapshots are read from.

    Mirrors ``LakeIO``'s own ``{lake_uri}/{tenant}/{module}/...`` layout
    (``libs/maxxflow_features/lake.py``) and ``ModelArtifactStore``'s
    ``{tenant}/model-artifacts/{module}/...`` (``libs/maxxflow_mlops/
    model_artifact_store.py``). This is the one place that convention is
    spelled out for M3 snapshots, so a future writer and this reader can
    never drift apart — see the module docstring for why none exists yet.
    """
    settings = settings or get_settings()
    base = settings.lake_uri.rstrip("/")
    return f"{base}/{tenant_id}/{SNAPSHOT_MODULE}/snapshots"


def _backend_label(lake_uri: str) -> str:
    """Cosmetic classification for log lines only — never a control-flow
    branch. The actual backend selection is entirely fsspec's own, driven by
    ``LAKE_URI``'s scheme (``Settings.lake_storage_options`` already supplies
    the matching credentials for either); this function does not participate
    in that dispatch, it only names it for observability.
    """
    if lake_uri.startswith("s3://"):
        return "minio_s3"
    if lake_uri.startswith(("abfs://", "abfss://")):
        return "azure_adls"
    if "://" in lake_uri:
        return lake_uri.split("://", 1)[0]
    return "unknown"


class LakeSnapshotSource:
    """Production default: lists/reads ``m3.snapshot.v1`` objects from the
    same medallion lake every other module uses — ``LAKE_URI`` +
    ``Settings.lake_storage_options`` — MinIO locally (``s3://``), Azure Data
    Lake in the cloud (``abfss://``), same code path, same credential
    resolution as ``ModelArtifactStore._filesystem`` (``libs/maxxflow_mlops/
    model_artifact_store.py``). No separate boto3/Azure-SDK code, no new
    authentication approach, no ``if backend == ...`` branch: fsspec picks
    the implementation from the URI scheme, exactly as the rest of this
    codebase already does.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def list_snapshots(self, tenant_id: str) -> list[dict[str, Any]]:
        prefix = snapshot_prefix(tenant_id, settings=self.settings)
        backend = _backend_label(self.settings.lake_uri)
        options = self.settings.lake_storage_options or None
        try:
            fs, root = fsspec.core.url_to_fs(prefix, **(options or {}))
            paths = fs.glob(f"{root.rstrip('/')}/*.json")
        except Exception as exc:
            # Storage unreachable/misconfigured degrades to "no history for
            # this tenant" — never a Weight Agent failure. Never log `options`
            # (carries LAKE_SECRET / the Azure connection string).
            log.warning(
                "snapshot_digest tenant=%s backend=%s prefix=%s list_failed=%s",
                tenant_id, backend, prefix, exc,
            )
            return []

        snapshots: list[dict[str, Any]] = []
        skipped = 0
        for path in paths:
            try:
                with fs.open(path, "rb") as handle:
                    record = json.loads(handle.read())
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("snapshot_digest: skipping unreadable object %s: %s", path, exc)
                skipped += 1
                continue
            if not isinstance(record, dict) or record.get("schema") != SNAPSHOT_SCHEMA:
                log.warning("snapshot_digest: skipping %s: not a %s record", path, SNAPSHOT_SCHEMA)
                skipped += 1
                continue
            snapshots.append(record)

        log.info(
            "snapshot_digest tenant=%s backend=%s prefix=%s objects_found=%d "
            "valid_snapshots=%d skipped_snapshots=%d",
            tenant_id, backend, prefix, len(paths), len(snapshots), skipped,
        )
        return snapshots


class LocalDirectorySnapshotSource:
    """Test/demo fixture override — never the production default. Reads
    every ``*.json`` file in ``directory`` regardless of tenant; tenant
    isolation is enforced the same way it always has been for this source,
    by :func:`build_snapshot_digest`'s own ``tenant`` field filter (this
    directory is a flat, mixed-tenant fixture, unlike the lake's per-tenant
    prefix, which isolates by path instead).
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def list_snapshots(self, tenant_id: str) -> list[dict[str, Any]]:
        return load_snapshots(self.directory)


def build_tenant_snapshot_digest(
    tenant_id: str,
    *,
    snapshot_source: SnapshotSource | None = None,
    directory: Path | None = None,
    max_snapshots: int = 50,
    settings: Settings | None = None,
) -> str | None:
    """Build ``tenant_id``'s history digest.

    Production default (``snapshot_source`` and ``directory`` both omitted):
    reads through :class:`LakeSnapshotSource` — MinIO locally, Azure Data
    Lake in the cloud, selected purely by ``LAKE_URI``, same as every other
    module. Pass ``directory`` for a local fixture/test override (e.g. the
    CLI's ``--snapshot-dir``), or ``snapshot_source`` for full control (e.g.
    a test double). Any storage failure — from the source itself, or raised
    here — degrades to ``None`` (no digest), the same "continue without this
    evidence" fallback an unreachable LLM provider already gets elsewhere in
    this module; it must never make M3 scoring unavailable.
    """
    if snapshot_source is None:
        snapshot_source = (
            LocalDirectorySnapshotSource(directory)
            if directory is not None
            else LakeSnapshotSource(settings)
        )
    try:
        snapshots = snapshot_source.list_snapshots(tenant_id)
    except Exception as exc:
        log.warning("snapshot_digest: snapshot source failed tenant=%s: %s", tenant_id, exc)
        snapshots = []

    digest = build_snapshot_digest(snapshots, tenant_id, max_snapshots=max_snapshots)
    log.info(
        "snapshot_digest tenant=%s objects_considered=%d digest_generated=%s",
        tenant_id, len(snapshots), digest is not None,
    )
    return digest
