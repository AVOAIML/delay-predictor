"""Operator-PII pseudonymization (plan §1a operator row, §12a #9).

``WorkOrder.assignedOperators`` / ``WorkOrderTimeLog.operatorId`` /
``Quotation.salesPersonId`` are raw ``public.users`` UUIDs — PII, and cross-schema.
They must NEVER reach silver / MLflow / an out-of-AU LLM.

So in the DAL, *before bronze*, every operator UUID is replaced by
``HMAC-SHA256(salt, uuid)`` -> a stable pseudonymous token. The salt is pinned in
the local secrets backend (and Key Vault in prod) so the same UUID maps to the
same token in local and prod — tiers are reproducible across environments.

The only operator-derived feature M3 keeps is an integer **skill tier**, computed
from that operator's *historical* real/expected duration ratios, keyed by the
pseudonymous token (never the UUID).
"""

from __future__ import annotations

import hashlib
import hmac

_TIER_COUNT = 5  # ordinal tiers 0 (slowest) .. 4 (fastest); 2 = unknown/median


def pseudonymize(raw_id: str | None, salt: str) -> str | None:
    """Stable HMAC-SHA256 token for an operator/user UUID. None passes through."""
    if raw_id is None or raw_id == "":
        return None
    digest = hmac.new(salt.encode("utf-8"), str(raw_id).encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()


def hash_to_bucket(token: str | None, n_buckets: int = _TIER_COUNT) -> int:
    """Deterministic fallback bucket from a token (when no performance history)."""
    if not token:
        return n_buckets // 2  # unknown -> median tier
    return int(token[:8], 16) % n_buckets


def skill_tier_from_ratio(real_expected_ratio: float | None,
                          n_tiers: int = _TIER_COUNT) -> int:
    """Map a historical real/expected duration ratio to an ordinal skill tier.

    ratio < 1  => faster than expected (higher tier). Unknown => median tier.
    Boundaries are fixed so tiers are reproducible local==prod.
    """
    if real_expected_ratio is None:
        return n_tiers // 2
    r = float(real_expected_ratio)
    # Fixed cut points on the real/expected ratio (lower is better/faster).
    if r <= 0.80:
        return 4
    if r <= 0.95:
        return 3
    if r <= 1.10:
        return 2
    if r <= 1.30:
        return 1
    return 0
