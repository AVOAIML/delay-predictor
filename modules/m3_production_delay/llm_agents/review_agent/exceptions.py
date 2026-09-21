"""Typed exceptions for the Review Agent, subclassing the platform base
(``MaXXFlowError``) rather than introducing a parallel hierarchy — same
convention as ``weight_agent/exceptions.py``."""

from __future__ import annotations

from maxxflow_core.errors import MaXXFlowError


class ReviewAgentError(MaXXFlowError):
    """Base for all Review Agent errors."""


class ReviewConfigError(ReviewAgentError):
    """Raised at config load time when a judging parameter is unusable (a
    non-positive token budget, a zero attempt budget, a temperature outside
    the provider-accepted range)."""


class JudgeResponseError(ReviewAgentError):
    """Raised internally when a judge response cannot be trusted — invalid
    JSON, a schema mismatch, a line index that names no candidate, a
    duplicate or incomplete verdict.

    Always caught inside :meth:`ReviewAgent.judge` and converted into an
    unapproved :class:`JudgeVerdict` carrying ``parse_error``; it never
    propagates to the pipeline. An unreadable answer must degrade to the
    deterministic template, not fail the batch.
    """
