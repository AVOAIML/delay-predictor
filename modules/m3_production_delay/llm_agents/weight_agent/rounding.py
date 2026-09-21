"""The one shared canonical-order remainder-assignment primitive (spec §5
step 5). Every place that must land an integer basis-point vector on an exact
target sum — projection, the historical blend, availability redistribution —
calls this instead of re-implementing its own rounding.

Determinism comes entirely from always walking ``order`` from the front.
"""

from __future__ import annotations

from collections.abc import Sequence


def ensure_finite_ints(values: dict[str, int], context: str) -> None:
    """No NaN/Infinity accepted anywhere in the numerical core (Improvement 6).
    Basis points are ``int`` by contract, so this rejects anything that isn't
    — a ``bool`` (a ``int`` subclass) is rejected too, since ``True``/``False``
    silently behaving as ``1``/``0`` in a weight vector would be a bug, not a
    feature."""
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{context}[{key!r}]={value!r} is not an integer")


def assign_remainder_canonical(
    values: dict[str, int],
    residual: int,
    order: Sequence[str],
    capacity: dict[str, int] | None = None,
) -> dict[str, int]:
    """Distribute ``residual`` (positive or negative) one unit at a time
    across ``values``, cycling ``order`` from the front, until consumed.

    ``capacity[signal]`` (if given) caps how many more units that signal may
    absorb in this call; a signal at zero remaining capacity is skipped. Pass
    ``capacity=None`` for the unconstrained case (blend, redistribution),
    where there is no per-signal ceiling on the remainder itself.

    Raises ``ValueError`` if capacity is supplied and is insufficient to
    absorb the full residual — callers that size ``capacity`` from the same
    headroom used to compute ``residual`` (see projection.py) never hit this;
    it exists as a defensive invariant check, not a normal runtime path.
    """
    ensure_finite_ints(values, "values")
    if not isinstance(residual, int) or isinstance(residual, bool):
        raise ValueError(f"residual={residual!r} is not an integer")
    if capacity is not None:
        ensure_finite_ints(capacity, "capacity")

    if residual == 0:
        return dict(values)
    if not order:
        raise ValueError("order must be non-empty when a residual remains")

    result = dict(values)
    remaining_capacity = dict(capacity) if capacity is not None else None
    sign = 1 if residual > 0 else -1
    remaining = abs(residual)
    n = len(order)
    index = 0
    stalled = 0
    while remaining > 0:
        signal = order[index % n]
        index += 1
        if remaining_capacity is not None:
            if remaining_capacity.get(signal, 0) <= 0:
                stalled += 1
                if stalled >= n:
                    raise ValueError(
                        "assign_remainder_canonical: insufficient capacity to "
                        f"absorb a residual of {remaining} more unit(s)"
                    )
                continue
            remaining_capacity[signal] -= 1
        result[signal] = result.get(signal, 0) + sign
        remaining -= 1
        stalled = 0
    return result
