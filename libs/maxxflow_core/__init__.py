"""maxxflow_core — shared primitives imported by every module and lib.

Contains the parity-critical building blocks that the plan (§1a, §12a) says
must be written ONCE and reused everywhere:

* settings   — pydantic-settings profiles (env-swapped; no ``if env==`` anywhere)
* clock      — single as-of clock (TZ-naive storage, Australia/Sydney presentation)
* money      — ``decimal.Decimal`` helpers for every margin / clamp / overrun boundary
* hashing    — HMAC(salt) -> skill-tier integer (operator UUIDs are PII)
* ports      — Protocol interfaces (the ports of ports/adapters)
* masterdata — canonical MasterData category + code constants (labels are UUIDs)
"""

from maxxflow_core.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
