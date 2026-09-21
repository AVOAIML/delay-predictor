"""Typed exceptions for the Weight Agent, subclassing the platform base
(``MaXXFlowError``) rather than introducing a parallel hierarchy."""

from __future__ import annotations

from maxxflow_core.errors import MaXXFlowError


class WeightAgentError(MaXXFlowError):
    """Base for all Weight Agent errors."""


class WeightConfigError(WeightAgentError):
    """Raised at config load time when the prior/bounds are infeasible."""


class NonZeroSumAdjustmentError(WeightAgentError):
    """Raised when an adjustment vector does not sum to zero (§5 step 1) — a
    hard validation failure, never silently renormalised away."""


class ProjectionInvariantError(WeightAgentError):
    """Raised if a converged projection fails sum==10000 or its own bounds.

    Convergence failure (residual != 0) is a normal runtime path (§5 step 6)
    that falls back to the prior — it never raises. This is different: it
    means the projection *claimed* to converge but the result is wrong, which
    is a bug in this module, not a runtime condition callers should handle.
    """


class LLMAdjustmentError(WeightAgentError):
    """Raised internally when either LLM stage fails validation (invalid
    JSON, schema mismatch, out-of-enum value, non-zero-sum adjustment, low
    profile confidence, missing critical profile fields). Always caught
    inside the resolver and converted into a ``fallback_reasons`` entry —
    never propagates out of ``WeightAgent.resolve``.
    """


class AllSignalsUnavailableError(WeightAgentError):
    """Raised when every one of the five signals is unavailable. There is no
    valid weight vector in this state — the old behaviour of returning an
    all-zero ``WeightResolution`` was a silent zero-risk-forever failure mode
    dressed up as a normal recommendation (a risk engine multiplying signals
    by these weights would score every work order as zero risk regardless of
    its actual signals). Unlike every other failure in this module, this one
    is NOT swallowed into ``fallback_reasons`` — there is no prior, blend, or
    LLM path that can produce a meaningful weight vector with zero inputs, so
    the caller (the orchestrator) must decide explicitly what "cannot score
    this work order" means for it, rather than being handed a result that
    looks resolved but isn't."""


class FittedWeightsError(WeightAgentError):
    """Raised when a :class:`FittedWeightsProvider` returns something that
    fails validation. Caught by the resolver and treated exactly like fitted
    weights being absent (§4.3 continues to the LLM path) — a misbehaving
    provider must not crash M3's advisory pipeline any more than an LLM
    failure would (Improvement 8)."""
