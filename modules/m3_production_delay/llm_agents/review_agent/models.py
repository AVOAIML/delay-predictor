"""The Review Agent's output contract — what the judge is allowed to say.

The agent owns this type, the way the Weight Agent owns ``WeightResolution``:
the consumer (``review/``) imports it, not the other way round, so the agent
never depends on the deterministic package it judges for.

Every field here is a boolean, an index, or a short diagnostic string. There
is deliberately nothing an LLM could use to introduce a number, a signal
value, or replacement prose — the judge can approve lines, or cause them to be
dropped, and nothing else. That is enforced by the shape of this type rather
than by prompt wording.

Same convention as ``weight_agent/models.py``: frozen dataclasses that
validate themselves in ``__post_init__``, so an invalid verdict is not
constructible and no caller has to remember a separate check.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Diagnostic strings are truncated at this length before being stored. A
#: judge that ignores the "short string" instruction must not be able to
#: smuggle prose into a payload that renders on screen.
MAX_DIAGNOSTIC_CHARS = 300

#: Upper bound on the diagnostic lists. Same reasoning, applied to length.
MAX_DIAGNOSTIC_ITEMS = 20


def _short_text(value: object, limit: int = MAX_DIAGNOSTIC_CHARS) -> str | None:
    if value is None:
        return None
    return str(value)[:limit]


def coerce_diagnostics(value: object, limit: int = MAX_DIAGNOSTIC_ITEMS) -> tuple[str, ...]:
    """Normalise a judge-supplied list of short strings: anything that is not
    a list becomes empty, non-scalar items are dropped, and both the item
    count and each item's length are capped. Public because the resolver
    applies it to the same fields before construction."""
    if not isinstance(value, list):
        return ()
    return tuple(
        str(item)[:MAX_DIAGNOSTIC_CHARS]
        for item in value
        if isinstance(item, (str, int, float))
    )[:limit]


@dataclass(frozen=True)
class JudgeLineVerdict:
    """The judge's answer for ONE candidate line."""

    line_index: int
    supported: bool
    evidence_ref: str | None = None
    issue: str | None = None

    def __post_init__(self) -> None:
        if self.line_index < 0:
            raise ValueError("line_index must be >= 0")
        if not isinstance(self.supported, bool):
            raise TypeError("supported must be a bool")

    def to_dict(self) -> dict:
        return {
            "line_index": self.line_index,
            "supported": self.supported,
            "evidence_ref": self.evidence_ref,
            "issue": self.issue,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "JudgeLineVerdict":
        return cls(
            line_index=payload["line_index"],
            supported=bool(payload["supported"]),
            evidence_ref=_short_text(payload.get("evidence_ref")),
            issue=_short_text(payload.get("issue")),
        )


@dataclass(frozen=True)
class JudgeVerdict:
    """The judge's whole response.

    ``parse_error`` is set when the model returned something that is not a
    valid verdict at all, or when the provider could not be reached. In that
    case ``approved`` is always False — an unreadable or missing answer is
    never assent, which ``__post_init__`` enforces rather than trusts.

    ``skipped`` marks the two cases where no call was made: a draft with no
    candidate lines (nothing to judge), and the agent being switched off.
    Recorded explicitly so "nobody asked" stays distinguishable from "the
    judge agreed".
    """

    approved: bool
    lines: tuple[JudgeLineVerdict, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
    omitted_signals: tuple[str, ...] = ()
    parse_error: str | None = None
    skipped: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.approved, bool):
            raise TypeError("approved must be a bool")
        for name, expected in (
            ("lines", JudgeLineVerdict),
            ("unsupported_claims", str),
            ("omitted_signals", str),
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be a tuple")
            for item in values:
                if not isinstance(item, expected):
                    raise TypeError(f"{name} must contain only {expected.__name__}")
        if self.parse_error and self.approved:
            raise ValueError("a verdict that failed to parse cannot be approved")
        indices = [line.line_index for line in self.lines]
        if len(indices) != len(set(indices)):
            raise ValueError("duplicate line_index in verdict")

    @property
    def unsupported_indices(self) -> tuple[int, ...]:
        return tuple(line.line_index for line in self.lines if not line.supported)

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "lines": [line.to_dict() for line in self.lines],
            "unsupported_claims": list(self.unsupported_claims),
            "omitted_signals": list(self.omitted_signals),
            "parse_error": self.parse_error,
            "skipped": self.skipped,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "JudgeVerdict":
        return cls(
            approved=bool(payload["approved"]),
            lines=tuple(JudgeLineVerdict.from_dict(line) for line in payload.get("lines") or ()),
            unsupported_claims=coerce_diagnostics(payload.get("unsupported_claims")),
            omitted_signals=coerce_diagnostics(payload.get("omitted_signals")),
            parse_error=_short_text(payload.get("parse_error")),
            skipped=bool(payload.get("skipped", False)),
        )


#: No call was made because there was nothing to judge. Shared and immutable,
#: so call sites need no Optional handling — same pattern as
#: ``weight_agent.tracing.NOOP_TRACER``.
SKIPPED_VERDICT = JudgeVerdict(approved=True, skipped=True)
