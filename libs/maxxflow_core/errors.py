"""Domain exceptions + a small logging helper (used across libs/modules)."""

from __future__ import annotations

import logging
import os


class MaXXFlowError(Exception):
    """Base for all platform errors."""


class LeakageError(MaXXFlowError):
    """Raised when an as-of-T leakage invariant is violated (M3) or a feature's
    single-feature AUC exceeds the alarm threshold (synthetic gate C)."""


class GuardrailError(MaXXFlowError):
    """Raised when a serving guardrail precondition is impossible to satisfy."""


class RegistryRoutingError(MaXXFlowError):
    """Raised if code attempts a tag/alias *search* (unsupported on Azure ML)."""


class TenantIsolationError(MaXXFlowError):
    """Raised if a query references a tenant_id column or crosses schemas."""


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger
