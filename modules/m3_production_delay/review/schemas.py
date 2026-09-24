"""Domain types for the Review Agent (Section 3) — the evidence the judge is
allowed to see, the candidate explanation it judges, its verdict, and the
validated insight that reaches the MO "AI Insights" panel.

Same convention as ``llm_agents/weight_agent/models.py``: every type is a
frozen dataclass that validates itself in ``__post_init__``, so an invalid
object is simply not constructible and no caller has to remember a separate
validation step. ``to_dict()``/``from_dict()`` are the JSON boundary —
``to_dict()`` output is always ``json.dumps(..., default=json_default)``-safe.

Two rules this module enforces structurally rather than by convention:

  * **No non-finite number is ever constructible.** ``material_shortfall_ratio``
    is ``math.inf`` whenever a short component has zero stock (see
    ``rule_engine/elements.py``), and that infinity propagates into
    ``composite_risk_score``. ``json.dumps`` would happily emit a bare
    ``Infinity`` literal, which is not JSON any consumer can parse. Every
    numeric field here goes through :func:`finite_or_none`, so the coercion
    happens once, at construction, instead of being re-checked at each
    serialisation site. ``evidence.py`` raises the matching
    ``Issue(check="non_finite")`` so the coercion is visible rather than
    silent.

  * **Every number in a line is traceable.** :class:`InsightLine` carries the
    evidence values it quoted (``quoted``) alongside the rendered text, which
    is what lets ``validators.check_quoted_numbers`` re-derive every numeral
    in the prose from :class:`EvidencePack` instead of trusting it.

The LLM never constructs any of these except :class:`JudgeVerdict`, and that
one only ever carries booleans, indices and short diagnostic strings — never
a number, a signal value, or a line of user-facing text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

# The verdict types belong to the Review Agent, which owns its own output
# contract exactly as the Weight Agent owns WeightResolution. They are
# re-exported here so a consumer of this module needs one import, not two —
# the dependency runs review -> llm_agents.review_agent and never back.
from m3_production_delay.llm_agents.review_agent.models import (  # noqa: F401
    SKIPPED_VERDICT,
    JudgeLineVerdict,
    JudgeVerdict,
)

# --- vocabularies -----------------------------------------------------------

#: Rule-engine signal keys this agent understands. The first four are the
#: tenant-weighted inputs to the base score. The critical-path cascade is a
#: deterministic score overlay. The raw predecessor ratio is unweighted
#: context and cannot appear as a cause by itself.
SIGNAL_TIME_OVERRUN = "time_overrun_ratio"
SIGNAL_OPERATOR_PACE = "operator_pace_ratio"
SIGNAL_MATERIAL_SHORTFALL = "material_shortfall_ratio"
SIGNAL_SUPPLIER_RELIABILITY = "supplier_reliability"
SIGNAL_PREDECESSOR_OVERRUN = "predecessor_time_overrun_ratio"
SIGNAL_CRITICAL_PATH_CASCADE = "critical_path_cascade_ratio"

#: Canonical order. Normative for every deterministic tie-break (line
#: ordering, omission checks) so two runs over the same evidence agree.
SIGNAL_ORDER: tuple[str, ...] = (
    SIGNAL_TIME_OVERRUN,
    SIGNAL_OPERATOR_PACE,
    SIGNAL_MATERIAL_SHORTFALL,
    SIGNAL_SUPPLIER_RELIABILITY,
    SIGNAL_CRITICAL_PATH_CASCADE,
    SIGNAL_PREDECESSOR_OVERRUN,
)
SIGNAL_SET: frozenset[str] = frozenset(SIGNAL_ORDER)

#: Signals whose evidence is Manufacturing-Order-wide rather than per
#: operation. ``rollup.py`` attaches the whole ``mo_components`` list to every
#: operation of a job, so these two carry the identical value on each
#: operation; emitting their line once per operation would repeat the same
#: sentence N times. See ``evidence.build_evidence`` for the de-duplication.
JOB_SCOPED_SIGNALS: frozenset[str] = frozenset(
    {SIGNAL_MATERIAL_SHORTFALL, SIGNAL_SUPPLIER_RELIABILITY}
)

SCOPE_OPERATION = "operation"
SCOPE_JOB = "job"
VALID_SCOPES: frozenset[str] = frozenset({SCOPE_OPERATION, SCOPE_JOB})
Scope = Literal["operation", "job"]

BASIS_QUANTITY = "quantity"
BASIS_OPERATOR_PACE = "operator_pace"
BASIS_NONE = "none"
VALID_OVERRUN_BASES: frozenset[str] = frozenset(
    {BASIS_QUANTITY, BASIS_OPERATOR_PACE, BASIS_NONE}
)
OverrunBasis = Literal["quantity", "operator_pace", "none"]

#: The only summary basis this version produces. Recorded explicitly because
#: the rule engine has no job-level roll-up at all: "the job's risk" here
#: means "its worst scorable operation", not an average and not a sum.
SUMMARY_BASIS_WORST_OPERATION = "worst_operation"

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"
VALID_SEVERITIES: frozenset[str] = frozenset(
    {SEVERITY_ERROR, SEVERITY_WARNING, SEVERITY_INFO}
)
Severity = Literal["error", "warning", "info"]

STATUS_APPROVED = "approved"
STATUS_APPROVED_WITH_WARNINGS = "approved_with_warnings"
STATUS_FALLBACK_TEMPLATE = "fallback_template"
STATUS_REJECTED = "rejected"
STATUS_SUPPRESSED_NOT_SCORABLE = "suppressed_not_scorable"
VALID_STATUSES: frozenset[str] = frozenset(
    {
        STATUS_APPROVED,
        STATUS_APPROVED_WITH_WARNINGS,
        STATUS_FALLBACK_TEMPLATE,
        STATUS_REJECTED,
        STATUS_SUPPRESSED_NOT_SCORABLE,
    }
)
InsightStatus = Literal[
    "approved",
    "approved_with_warnings",
    "fallback_template",
    "rejected",
    "suppressed_not_scorable",
]

#: Statuses that mean "the panel is not showing a fully validated
#: explanation" — every one of them gets an ``audit_logs`` row on publish,
#: the same way M1/M2 audit their suppressed and low-confidence outputs.
AUDITED_STATUSES: frozenset[str] = frozenset(
    {STATUS_FALLBACK_TEMPLATE, STATUS_REJECTED, STATUS_SUPPRESSED_NOT_SCORABLE}
)

SOURCE_TEMPLATE = "template"
VALID_SOURCES: frozenset[str] = frozenset({SOURCE_TEMPLATE})


def finite_or_none(value: Any) -> float | None:
    """``float(value)`` when it is a real, finite number; ``None`` otherwise.

    ``None`` in, ``None`` out. ``math.inf`` (zero-stock shortfall) and ``nan``
    both come back as ``None`` — callers that need to *report* the coercion
    compare before and after (``evidence.py`` does exactly that) rather than
    relying on a second, separate finiteness check.
    """
    if value is None:
        return None
    if isinstance(value, bool):  # bool is an int; never a measurement here
        raise TypeError(f"expected a number, got bool {value!r}")
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return as_float if math.isfinite(as_float) else None


def _require_finite(name: str, value: Any) -> float | None:
    """Field guard: the value must already have been coerced by the builder.

    Constructing an evidence object straight from a raw rule-engine number is
    the one way an infinity could still reach a payload, so that is a
    programming error here, not something to silently repair.
    """
    if value is None:
        return None
    coerced = finite_or_none(value)
    if coerced is None:
        raise ValueError(
            f"{name}={value!r} is not finite; coerce it with finite_or_none() and "
            "record an Issue(check='non_finite') before constructing this object"
        )
    return coerced


def _tuple_of(name: str, values: Any, expected: type) -> tuple:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple, got {type(values).__name__}")
    for item in values:
        if not isinstance(item, expected):
            raise TypeError(f"{name} must contain only {expected.__name__}")
    return values


# --- issues -----------------------------------------------------------------


@dataclass(frozen=True)
class Issue:
    """One deterministic finding about an insight — a validator failure, a
    plausibility warning, or a piece of context worth carrying (cascading
    predecessor). ``severity`` decides what the pipeline does with it:
    ``error`` rejects the draft, ``warning`` downgrades an approval,
    ``info`` is carried for the reader and nothing else.
    """

    check: str
    severity: str
    message: str
    ref: str | None = None  # operation_id / component_id / line index, when scoped

    def __post_init__(self) -> None:
        if not self.check:
            raise ValueError("check must not be empty")
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}")
        if not self.message:
            raise ValueError("message must not be empty")

    @property
    def is_error(self) -> bool:
        return self.severity == SEVERITY_ERROR

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
            "ref": self.ref,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Issue":
        return cls(
            check=payload["check"],
            severity=payload["severity"],
            message=payload["message"],
            ref=payload.get("ref"),
        )


def has_errors(issues: tuple[Issue, ...]) -> bool:
    return any(issue.is_error for issue in issues)


# --- evidence ---------------------------------------------------------------


@dataclass(frozen=True)
class SignalEvidence:
    """One rule-engine signal as the judge sees it: its value, the weight it
    carried into ``composite_risk_score``, whether it cleared its fire
    baseline, its share of the score, and the raw numbers a line may quote.

    ``contribution`` is a share of the final score. For tenant-weighted
    signals it accounts for the base composite's renormalisation; for the
    critical-path cascade it is the share added by the deterministic overlay.
    It is ``None`` whenever the share is not computable, never a fabricated
    zero, because "contributed nothing" and "cannot be computed" order
    differently.
    """

    key: str
    value: float | None
    weight: float
    fired: bool
    contribution: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.key not in SIGNAL_SET:
            raise ValueError(f"unknown signal {self.key!r}")
        object.__setattr__(self, "value", _require_finite(f"{self.key}.value", self.value))
        weight = _require_finite(f"{self.key}.weight", self.weight)
        if weight is None or weight < 0:
            raise ValueError(f"{self.key}.weight={self.weight!r} must be a finite value >= 0")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(
            self, "contribution", _require_finite(f"{self.key}.contribution", self.contribution)
        )
        if not isinstance(self.fired, bool):
            raise TypeError(f"{self.key}.fired must be a bool")
        if self.value is None and self.fired:
            raise ValueError(f"{self.key} cannot fire with no value")
        if not isinstance(self.detail, dict):
            raise TypeError(f"{self.key}.detail must be a dict")

    @property
    def is_weighted(self) -> bool:
        return self.weight > 0

    @property
    def affects_score(self) -> bool:
        """Whether this signal can truthfully be presented as a score cause.

        Most causes participate through a tenant weight. The critical-path
        cascade instead changes the score through a deterministic overlay;
        its positive contribution records that change without pretending it
        received a Weight Agent weight.
        """
        return self.is_weighted or (
            self.key == SIGNAL_CRITICAL_PATH_CASCADE
            and self.contribution is not None
            and self.contribution > 0
        )

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "weight": self.weight,
            "fired": self.fired,
            "contribution": self.contribution,
            "detail": dict(self.detail),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SignalEvidence":
        return cls(
            key=payload["key"],
            value=payload.get("value"),
            weight=payload["weight"],
            fired=bool(payload["fired"]),
            contribution=payload.get("contribution"),
            detail=dict(payload.get("detail") or {}),
        )


@dataclass(frozen=True)
class ComponentEvidence:
    """A component that is short in the warehouse, with the quantity short —
    never the shortfall *ratio*. The ratio is a modelling input whose
    magnitude is meaningless to a production planner (and is infinite at zero
    stock); "60 short" is the number that belongs on screen.
    """

    component_id: str
    name: str | None
    required_quantity: float | None
    available_quantity: float | None
    shortfall_quantity: float | None
    vendor_name: str | None = None
    vendor_lead_time_ratio: float | None = None

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("component_id must not be empty")
        for name in ("required_quantity", "available_quantity", "shortfall_quantity",
                     "vendor_lead_time_ratio"):
            object.__setattr__(self, name, _require_finite(name, getattr(self, name)))

    def to_dict(self) -> dict:
        return {
            "component_id": self.component_id,
            "name": self.name,
            "required_quantity": self.required_quantity,
            "available_quantity": self.available_quantity,
            "shortfall_quantity": self.shortfall_quantity,
            "vendor_name": self.vendor_name,
            "vendor_lead_time_ratio": self.vendor_lead_time_ratio,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ComponentEvidence":
        return cls(
            component_id=payload["component_id"],
            name=payload.get("name"),
            required_quantity=payload.get("required_quantity"),
            available_quantity=payload.get("available_quantity"),
            shortfall_quantity=payload.get("shortfall_quantity"),
            vendor_name=payload.get("vendor_name"),
            vendor_lead_time_ratio=payload.get("vendor_lead_time_ratio"),
        )


@dataclass(frozen=True)
class OperationEvidence:
    """One operation's whole evidence row.

    ``is_scorable`` is this agent's own gate (the user story's 25%-progress
    rule), not something the rule engine reports: it scores every operation
    regardless of progress, so an operation that has not started can still
    come back ``is_delayed=True`` on operator/material/supplier signals alone.
    A non-scorable operation contributes no explanation lines and no badge —
    its engine numbers are kept, under ``engine`` in the serialised form, so
    the panel can still show them without this agent vouching for them.
    """

    operation_id: str
    operation_name: str
    status: str | None
    expected_duration_minutes: float | None
    actual_duration_minutes: float | None
    job_quantity: float | None
    current_done_quantity: float | None
    composite_risk_score: float | None
    is_delayed: bool | None
    predicted_overrun_hours: float | None
    overrun_basis: str
    signals: tuple[SignalEvidence, ...]
    components_short: tuple[ComponentEvidence, ...]
    depends_on_operation_ids: tuple[str, ...]
    is_scorable: bool

    def __post_init__(self) -> None:
        if not self.operation_id:
            raise ValueError("operation_id must not be empty")
        if not self.operation_name:
            raise ValueError("operation_name must not be empty")
        for name in ("expected_duration_minutes", "actual_duration_minutes", "job_quantity",
                     "current_done_quantity", "composite_risk_score", "predicted_overrun_hours"):
            object.__setattr__(self, name, _require_finite(name, getattr(self, name)))
        if self.overrun_basis not in VALID_OVERRUN_BASES:
            raise ValueError(f"unknown overrun_basis {self.overrun_basis!r}")
        if self.is_delayed is not None and not isinstance(self.is_delayed, bool):
            raise TypeError("is_delayed must be a bool or None")
        if not isinstance(self.is_scorable, bool):
            raise TypeError("is_scorable must be a bool")
        _tuple_of("signals", self.signals, SignalEvidence)
        _tuple_of("components_short", self.components_short, ComponentEvidence)
        _tuple_of("depends_on_operation_ids", self.depends_on_operation_ids, str)
        keys = [signal.key for signal in self.signals]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate signal keys on operation {self.operation_id!r}")

    def signal(self, key: str) -> SignalEvidence | None:
        for candidate in self.signals:
            if candidate.key == key:
                return candidate
        return None

    def to_dict(self) -> dict:
        return {
            "operation_id": self.operation_id,
            "operation_name": self.operation_name,
            "status": self.status,
            "expected_duration_minutes": self.expected_duration_minutes,
            "actual_duration_minutes": self.actual_duration_minutes,
            "job_quantity": self.job_quantity,
            "current_done_quantity": self.current_done_quantity,
            "composite_risk_score": self.composite_risk_score,
            "is_delayed": self.is_delayed,
            "predicted_overrun_hours": self.predicted_overrun_hours,
            "overrun_basis": self.overrun_basis,
            "signals": [signal.to_dict() for signal in self.signals],
            "components_short": [c.to_dict() for c in self.components_short],
            "depends_on_operation_ids": list(self.depends_on_operation_ids),
            "is_scorable": self.is_scorable,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "OperationEvidence":
        return cls(
            operation_id=payload["operation_id"],
            operation_name=payload["operation_name"],
            status=payload.get("status"),
            expected_duration_minutes=payload.get("expected_duration_minutes"),
            actual_duration_minutes=payload.get("actual_duration_minutes"),
            job_quantity=payload.get("job_quantity"),
            current_done_quantity=payload.get("current_done_quantity"),
            composite_risk_score=payload.get("composite_risk_score"),
            is_delayed=payload.get("is_delayed"),
            predicted_overrun_hours=payload.get("predicted_overrun_hours"),
            overrun_basis=payload["overrun_basis"],
            signals=tuple(SignalEvidence.from_dict(s) for s in payload.get("signals") or ()),
            components_short=tuple(
                ComponentEvidence.from_dict(c) for c in payload.get("components_short") or ()
            ),
            depends_on_operation_ids=tuple(payload.get("depends_on_operation_ids") or ()),
            is_scorable=bool(payload["is_scorable"]),
        )


@dataclass(frozen=True)
class EvidencePack:
    """Everything, and only what, the judge is allowed to reason over.

    ``delay_threshold`` and ``weights`` are carried explicitly because neither
    is discoverable from a scored job: the same composite score means
    "delayed" or "fine" depending on the tenant's threshold, and a signal's
    contribution is meaningless without the weight vector that produced it.

    The summary fields are this agent's own job-level aggregation — the rule
    engine produces per-operation results only. ``summary_basis`` names the
    aggregation so no reader has to infer whether a number is a max, a mean or
    a sum.
    """

    job_id: str
    delay_threshold: float
    weights: dict[str, float]
    operations: tuple[OperationEvidence, ...]
    material_overrun: tuple[ComponentEvidence, ...]
    job_signals: tuple[SignalEvidence, ...]
    summary_risk_score: float | None
    summary_overrun_hours: float | None
    summary_is_delayed: bool | None
    summary_basis: str
    issues: tuple[Issue, ...] = ()

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job_id must not be empty")
        threshold = _require_finite("delay_threshold", self.delay_threshold)
        if threshold is None:
            raise ValueError("delay_threshold is required and must be a finite number")
        object.__setattr__(self, "delay_threshold", threshold)
        if not isinstance(self.weights, dict):
            raise TypeError("weights must be a dict")
        for key, weight in self.weights.items():
            if _require_finite(f"weights[{key!r}]", weight) is None:
                raise ValueError(f"weights[{key!r}] must be a number")
        for name in ("summary_risk_score", "summary_overrun_hours"):
            object.__setattr__(self, name, _require_finite(name, getattr(self, name)))
        if self.summary_is_delayed is not None and not isinstance(self.summary_is_delayed, bool):
            raise TypeError("summary_is_delayed must be a bool or None")
        _tuple_of("operations", self.operations, OperationEvidence)
        _tuple_of("material_overrun", self.material_overrun, ComponentEvidence)
        _tuple_of("job_signals", self.job_signals, SignalEvidence)
        _tuple_of("issues", self.issues, Issue)
        for signal in self.job_signals:
            if signal.key not in JOB_SCOPED_SIGNALS:
                raise ValueError(
                    f"{signal.key!r} is not a job-scoped signal; job_signals may only carry "
                    f"{sorted(JOB_SCOPED_SIGNALS)}"
                )

    @property
    def scorable_operations(self) -> tuple[OperationEvidence, ...]:
        return tuple(op for op in self.operations if op.is_scorable)

    def operation(self, operation_id: str) -> OperationEvidence | None:
        for op in self.operations:
            if op.operation_id == operation_id:
                return op
        return None

    def job_signal(self, key: str) -> SignalEvidence | None:
        for signal in self.job_signals:
            if signal.key == key:
                return signal
        return None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "delay_threshold": self.delay_threshold,
            "weights": dict(self.weights),
            "operations": [op.to_dict() for op in self.operations],
            "material_overrun": [c.to_dict() for c in self.material_overrun],
            "job_signals": [signal.to_dict() for signal in self.job_signals],
            "summary_risk_score": self.summary_risk_score,
            "summary_overrun_hours": self.summary_overrun_hours,
            "summary_is_delayed": self.summary_is_delayed,
            "summary_basis": self.summary_basis,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "EvidencePack":
        return cls(
            job_id=payload["job_id"],
            delay_threshold=payload["delay_threshold"],
            weights=dict(payload.get("weights") or {}),
            operations=tuple(
                OperationEvidence.from_dict(op) for op in payload.get("operations") or ()
            ),
            material_overrun=tuple(
                ComponentEvidence.from_dict(c) for c in payload.get("material_overrun") or ()
            ),
            job_signals=tuple(
                SignalEvidence.from_dict(s) for s in payload.get("job_signals") or ()
            ),
            summary_risk_score=payload.get("summary_risk_score"),
            summary_overrun_hours=payload.get("summary_overrun_hours"),
            summary_is_delayed=payload.get("summary_is_delayed"),
            summary_basis=payload.get("summary_basis", SUMMARY_BASIS_WORST_OPERATION),
            issues=tuple(Issue.from_dict(i) for i in payload.get("issues") or ()),
        )


# --- draft ------------------------------------------------------------------


@dataclass(frozen=True)
class InsightLine:
    """One explanation line, in the user story's three-part shape: a headline
    naming the cause, a detail naming the subject and the numbers, and an
    optional delta.

    ``quoted`` is the contract that makes the whole design checkable: it maps
    each numeral rendered into the text back to the evidence field it came
    from, so ``validators.check_quoted_numbers`` can re-derive the prose
    instead of trusting it, and the judge is only ever asked whether the
    *claim* is supported — never to produce or verify arithmetic.
    """

    index: int
    signal_key: str
    scope: str
    headline: str
    detail: str
    delta: str | None = None
    operation_id: str | None = None
    contribution: float | None = None
    quoted: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("index must be >= 0")
        if self.signal_key not in SIGNAL_SET:
            raise ValueError(f"unknown signal {self.signal_key!r}")
        if self.scope not in VALID_SCOPES:
            raise ValueError(f"unknown scope {self.scope!r}")
        if not self.headline or not self.detail:
            raise ValueError("headline and detail must not be empty")
        if self.scope == SCOPE_OPERATION and not self.operation_id:
            raise ValueError("an operation-scoped line must name its operation_id")
        object.__setattr__(
            self, "contribution", _require_finite("contribution", self.contribution)
        )
        if not isinstance(self.quoted, dict):
            raise TypeError("quoted must be a dict")
        for name, value in self.quoted.items():
            if _require_finite(f"quoted[{name!r}]", value) is None:
                raise ValueError(f"quoted[{name!r}] must be a finite number")

    @property
    def text(self) -> str:
        return f"{self.headline} — {self.detail}" + (f" ({self.delta})" if self.delta else "")

    def renumbered(self, index: int) -> "InsightLine":
        return InsightLine(
            index=index,
            signal_key=self.signal_key,
            scope=self.scope,
            headline=self.headline,
            detail=self.detail,
            delta=self.delta,
            operation_id=self.operation_id,
            contribution=self.contribution,
            quoted=dict(self.quoted),
        )

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "signal_key": self.signal_key,
            "scope": self.scope,
            "headline": self.headline,
            "detail": self.detail,
            "delta": self.delta,
            "operation_id": self.operation_id,
            "contribution": self.contribution,
            "quoted": dict(self.quoted),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "InsightLine":
        return cls(
            index=payload["index"],
            signal_key=payload["signal_key"],
            scope=payload["scope"],
            headline=payload["headline"],
            detail=payload["detail"],
            delta=payload.get("delta"),
            operation_id=payload.get("operation_id"),
            contribution=payload.get("contribution"),
            quoted=dict(payload.get("quoted") or {}),
        )


@dataclass(frozen=True)
class InsightDraft:
    """The candidate insight, before any judging. ``source`` exists to make
    the provenance unambiguous in the payload: there is exactly one producer
    of these lines (the deterministic composer), and no code path lets an LLM
    write, rewrite or reorder one.
    """

    job_id: str
    summary_overrun_hours: float | None
    summary_risk_score: float | None
    summary_is_delayed: bool | None
    summary_basis: str
    why_lines: tuple[InsightLine, ...]
    material_overrun: tuple[ComponentEvidence, ...]
    source: str = SOURCE_TEMPLATE

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job_id must not be empty")
        for name in ("summary_overrun_hours", "summary_risk_score"):
            object.__setattr__(self, name, _require_finite(name, getattr(self, name)))
        if self.summary_is_delayed is not None and not isinstance(self.summary_is_delayed, bool):
            raise TypeError("summary_is_delayed must be a bool or None")
        _tuple_of("why_lines", self.why_lines, InsightLine)
        _tuple_of("material_overrun", self.material_overrun, ComponentEvidence)
        if self.source not in VALID_SOURCES:
            raise ValueError(f"unknown source {self.source!r}")
        for position, line in enumerate(self.why_lines):
            if line.index != position:
                raise ValueError(
                    f"why_lines must be indexed 0..n-1 in order; line at position {position} "
                    f"has index {line.index}"
                )

    def with_lines(self, lines: tuple[InsightLine, ...]) -> "InsightDraft":
        """A copy carrying ``lines``, re-indexed 0..n-1 — the judge's
        drop-and-retry step needs the indices to stay dense so a second
        verdict's ``line_index`` is unambiguous."""
        return InsightDraft(
            job_id=self.job_id,
            summary_overrun_hours=self.summary_overrun_hours,
            summary_risk_score=self.summary_risk_score,
            summary_is_delayed=self.summary_is_delayed,
            summary_basis=self.summary_basis,
            why_lines=tuple(line.renumbered(i) for i, line in enumerate(lines)),
            material_overrun=self.material_overrun,
            source=self.source,
        )

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "summary_overrun_hours": self.summary_overrun_hours,
            "summary_risk_score": self.summary_risk_score,
            "summary_is_delayed": self.summary_is_delayed,
            "summary_basis": self.summary_basis,
            "why_lines": [line.to_dict() for line in self.why_lines],
            "material_overrun": [c.to_dict() for c in self.material_overrun],
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "InsightDraft":
        return cls(
            job_id=payload["job_id"],
            summary_overrun_hours=payload.get("summary_overrun_hours"),
            summary_risk_score=payload.get("summary_risk_score"),
            summary_is_delayed=payload.get("summary_is_delayed"),
            summary_basis=payload.get("summary_basis", SUMMARY_BASIS_WORST_OPERATION),
            why_lines=tuple(InsightLine.from_dict(line) for line in payload.get("why_lines") or ()),
            material_overrun=tuple(
                ComponentEvidence.from_dict(c) for c in payload.get("material_overrun") or ()
            ),
            source=payload.get("source", SOURCE_TEMPLATE),
        )


# --- output -----------------------------------------------------------------


@dataclass(frozen=True)
class ValidatedInsight:
    """What the MO "AI Insights" panel renders, and what ``publish.py`` writes
    to ``manufacturing_orders.customElements`` under ``ai_delay_insight``.

    ``material_overrun`` is intentionally independent of ``is_delayed``: a
    short component is a fact about the warehouse, worth showing whether or
    not the composite score crossed the threshold.
    """

    job_id: str
    status: str
    overrun_hours: float | None
    risk_score: float | None
    is_delayed: bool | None
    summary_basis: str
    delay_threshold: float
    why_lines: tuple[InsightLine, ...]
    material_overrun: tuple[ComponentEvidence, ...]
    issues: tuple[Issue, ...]
    judge: JudgeVerdict
    attempts: int
    model_version: str
    generated_at: str | None = None

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job_id must not be empty")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"unknown status {self.status!r}")
        for name in ("overrun_hours", "risk_score"):
            object.__setattr__(self, name, _require_finite(name, getattr(self, name)))
        threshold = _require_finite("delay_threshold", self.delay_threshold)
        if threshold is None:
            raise ValueError("delay_threshold is required and must be a finite number")
        object.__setattr__(self, "delay_threshold", threshold)
        if self.is_delayed is not None and not isinstance(self.is_delayed, bool):
            raise TypeError("is_delayed must be a bool or None")
        _tuple_of("why_lines", self.why_lines, InsightLine)
        _tuple_of("material_overrun", self.material_overrun, ComponentEvidence)
        _tuple_of("issues", self.issues, Issue)
        if not isinstance(self.judge, JudgeVerdict):
            raise TypeError("judge must be a JudgeVerdict")
        if self.attempts < 0:
            raise ValueError("attempts must be >= 0")
        if not self.model_version:
            raise ValueError("model_version must not be empty")
        if self.status in (STATUS_REJECTED, STATUS_SUPPRESSED_NOT_SCORABLE) and self.why_lines:
            raise ValueError(f"status={self.status!r} must carry no why_lines")

    @property
    def needs_audit(self) -> bool:
        return self.status in AUDITED_STATUSES

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "overrun_hours": self.overrun_hours,
            "risk_score": self.risk_score,
            "is_delayed": self.is_delayed,
            "summary_basis": self.summary_basis,
            "delay_threshold": self.delay_threshold,
            "why_lines": [line.to_dict() for line in self.why_lines],
            "material_overrun": [c.to_dict() for c in self.material_overrun],
            "issues": [issue.to_dict() for issue in self.issues],
            "judge": self.judge.to_dict(),
            "attempts": self.attempts,
            "model_version": self.model_version,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ValidatedInsight":
        return cls(
            job_id=payload["job_id"],
            status=payload["status"],
            overrun_hours=payload.get("overrun_hours"),
            risk_score=payload.get("risk_score"),
            is_delayed=payload.get("is_delayed"),
            summary_basis=payload.get("summary_basis", SUMMARY_BASIS_WORST_OPERATION),
            delay_threshold=payload["delay_threshold"],
            why_lines=tuple(InsightLine.from_dict(line) for line in payload.get("why_lines") or ()),
            material_overrun=tuple(
                ComponentEvidence.from_dict(c) for c in payload.get("material_overrun") or ()
            ),
            issues=tuple(Issue.from_dict(i) for i in payload.get("issues") or ()),
            judge=JudgeVerdict.from_dict(payload["judge"]),
            attempts=payload["attempts"],
            model_version=payload["model_version"],
            generated_at=payload.get("generated_at"),
        )
