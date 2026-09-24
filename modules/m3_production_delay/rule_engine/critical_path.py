"""Critical-path analysis and deterministic cascading-delay propagation.

The dependency rows in Prisma describe a directed acyclic graph where an
operation depends on its predecessors.  This module applies the standard
Critical Path Method (CPM) to that graph using each work order's expected
duration.  It deliberately has no database access and no LLM involvement.

Only an overrun travelling along a zero-float CPM edge becomes a cascade.
Consequently, an independent operation and a dependency on a non-critical
branch cannot raise another operation's risk.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


CASCADE_OVERRUN_BASELINE = 1.20
_FLOAT_TOLERANCE_MINUTES = 1e-7


@dataclass(frozen=True)
class CriticalPathNode:
    operation_id: str
    earliest_start_minutes: float
    earliest_finish_minutes: float
    latest_start_minutes: float
    latest_finish_minutes: float
    total_float_minutes: float
    is_critical: bool
    critical_predecessor_ids: tuple[str, ...]


@dataclass(frozen=True)
class CriticalPathResult:
    valid: bool
    project_duration_minutes: float
    topological_order: tuple[str, ...]
    nodes: dict[str, CriticalPathNode]
    error: str | None = None


@dataclass(frozen=True)
class CascadeSignal:
    ratio: float
    source_operation_ids: tuple[str, ...]


def _duration(op: dict[str, Any]) -> float:
    raw = op.get("expected_duration_minutes")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0 else 0.0


def analyze_critical_path(operations: list[dict]) -> CriticalPathResult:
    """Run a forward/backward CPM pass over one manufacturing order.

    Dependencies which name an operation outside this job are ignored.  A
    cyclic graph is returned as ``valid=False`` and produces no critical
    nodes; callers can safely decline cascade scoring without failing the
    rest of the job's risk calculation.
    """
    order_index: dict[str, int] = {}
    operations_by_id: dict[str, dict] = {}
    for index, op in enumerate(operations):
        operation_id = str(op.get("operation_id") or "")
        if not operation_id or operation_id in operations_by_id:
            return CriticalPathResult(
                valid=False,
                project_duration_minutes=0.0,
                topological_order=(),
                nodes={},
                error="operation ids must be non-empty and unique",
            )
        order_index[operation_id] = index
        operations_by_id[operation_id] = op

    if not operations_by_id:
        return CriticalPathResult(True, 0.0, (), {})

    predecessors: dict[str, list[str]] = {operation_id: [] for operation_id in operations_by_id}
    successors: dict[str, list[str]] = {operation_id: [] for operation_id in operations_by_id}
    for operation_id, op in operations_by_id.items():
        seen: set[str] = set()
        for raw_predecessor in op.get("depends_on_operation_ids") or ():
            predecessor_id = str(raw_predecessor)
            if predecessor_id not in operations_by_id or predecessor_id in seen:
                continue
            seen.add(predecessor_id)
            predecessors[operation_id].append(predecessor_id)
            successors[predecessor_id].append(operation_id)

    # Stable Kahn traversal: input order is only a deterministic tie-break;
    # it does not determine which path is critical.
    indegree = {operation_id: len(preds) for operation_id, preds in predecessors.items()}
    ready = sorted(
        (operation_id for operation_id, count in indegree.items() if count == 0),
        key=order_index.__getitem__,
    )
    topo: list[str] = []
    while ready:
        operation_id = ready.pop(0)
        topo.append(operation_id)
        for successor_id in sorted(successors[operation_id], key=order_index.__getitem__):
            indegree[successor_id] -= 1
            if indegree[successor_id] == 0:
                ready.append(successor_id)
                ready.sort(key=order_index.__getitem__)

    if len(topo) != len(operations_by_id):
        return CriticalPathResult(
            valid=False,
            project_duration_minutes=0.0,
            topological_order=tuple(topo),
            nodes={},
            error="operation dependency graph contains a cycle",
        )

    durations = {
        operation_id: _duration(operations_by_id[operation_id])
        for operation_id in operations_by_id
    }
    earliest_start: dict[str, float] = {}
    earliest_finish: dict[str, float] = {}
    for operation_id in topo:
        earliest_start[operation_id] = max(
            (earliest_finish[pred] for pred in predecessors[operation_id]), default=0.0
        )
        earliest_finish[operation_id] = earliest_start[operation_id] + durations[operation_id]

    project_duration = max(earliest_finish.values(), default=0.0)
    latest_start: dict[str, float] = {}
    latest_finish: dict[str, float] = {}
    for operation_id in reversed(topo):
        latest_finish[operation_id] = min(
            (latest_start[successor] for successor in successors[operation_id]),
            default=project_duration,
        )
        latest_start[operation_id] = latest_finish[operation_id] - durations[operation_id]

    critical = {
        operation_id: abs(latest_start[operation_id] - earliest_start[operation_id])
        <= _FLOAT_TOLERANCE_MINUTES
        for operation_id in topo
    }
    nodes: dict[str, CriticalPathNode] = {}
    for operation_id in topo:
        critical_predecessors = tuple(
            predecessor_id
            for predecessor_id in predecessors[operation_id]
            if critical[operation_id]
            and critical[predecessor_id]
            and abs(earliest_finish[predecessor_id] - earliest_start[operation_id])
            <= _FLOAT_TOLERANCE_MINUTES
        )
        nodes[operation_id] = CriticalPathNode(
            operation_id=operation_id,
            earliest_start_minutes=earliest_start[operation_id],
            earliest_finish_minutes=earliest_finish[operation_id],
            latest_start_minutes=latest_start[operation_id],
            latest_finish_minutes=latest_finish[operation_id],
            total_float_minutes=max(0.0, latest_start[operation_id] - earliest_start[operation_id]),
            is_critical=critical[operation_id],
            critical_predecessor_ids=critical_predecessors,
        )
    return CriticalPathResult(True, project_duration, tuple(topo), nodes)


def cascading_overruns(
    analysis: CriticalPathResult,
    own_overrun_ratios: dict[str, float | None],
    baseline: float = CASCADE_OVERRUN_BASELINE,
) -> dict[str, CascadeSignal | None]:
    """Return the worst inherited overrun on each critical-path operation.

    The signal is propagated through successive critical edges, so a late
    upstream operation can warn multiple not-yet-started successors.  The
    source id remains the operation whose own measured ratio crossed 1.20.
    """
    signals: dict[str, CascadeSignal | None] = {
        operation_id: None for operation_id in analysis.nodes
    }
    if not analysis.valid:
        return signals

    for operation_id in analysis.topological_order:
        node = analysis.nodes[operation_id]
        candidates: list[tuple[float, str]] = []
        for predecessor_id in node.critical_predecessor_ids:
            own_ratio = own_overrun_ratios.get(predecessor_id)
            if own_ratio is not None and math.isfinite(own_ratio) and own_ratio > baseline:
                candidates.append((own_ratio, predecessor_id))
            inherited = signals.get(predecessor_id)
            if inherited is not None:
                candidates.extend((inherited.ratio, source_id) for source_id in inherited.source_operation_ids)
        if not candidates:
            continue
        worst = max(ratio for ratio, _ in candidates)
        source_ids = tuple(sorted({source_id for ratio, source_id in candidates if ratio == worst}))
        signals[operation_id] = CascadeSignal(worst, source_ids)
    return signals


def apply_cascade_risk(base_score: float | None, cascade_ratio: float | None) -> float | None:
    """Apply a conservative, monotonic cascade overlay to a base score.

    The inherited excess above the on-plan baseline (``ratio - 1``) is added
    to an existing score.  For a dependent with no usable local evidence, the
    predecessor ratio itself is the floor.  Therefore a real cascade always
    raises an existing score and a not-started dependent can still be marked
    at risk.
    """
    if cascade_ratio is None:
        return base_score
    base = 0.0 if base_score is None else base_score
    return max(cascade_ratio, base + (cascade_ratio - 1.0))
