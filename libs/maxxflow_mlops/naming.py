"""Registered-model naming (plan §7, §12a #1).

The registered-model NAME is the routing key: ``t_<tenant_slug>__m_<module>``.
Routing is by name + ``@champion`` alias load ONLY — never tag/alias *search*
(unsupported on Azure ML's MLflow registry). One artifact per (tenant, module).
"""

from __future__ import annotations

_PREFIX_T = "t_"
_SEP = "__m_"

# The shared "base" tenant. Models registered under this slug (t_global__m_<module>)
# are trained by us (developers) on pooled data and are what every newly onboarded
# tenant SEES and is served, until that tenant trains + publishes its own champion.
GLOBAL_TENANT = "global"


def registered_model_name(tenant: str, module: str) -> str:
    return f"{_PREFIX_T}{tenant}{_SEP}{module}"


def global_model_name(module: str) -> str:
    """Registered name of the shared base model for a module."""
    return registered_model_name(GLOBAL_TENANT, module)


def parse_model_name(name: str) -> tuple[str, str]:
    if not name.startswith(_PREFIX_T) or _SEP not in name:
        raise ValueError(f"not a maxxflow model name: {name!r}")
    body = name[len(_PREFIX_T):]
    tenant, module = body.split(_SEP, 1)
    return tenant, module
