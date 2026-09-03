"""THE single BYOC scoring script (plan §2 serving row, §7, §12a #2).

This exact file runs under ``azureml-inference-server-http`` (azmlinfsrv) both
locally (compose ``model-server``) and on the Azure ML managed online endpoint —
so routes, request envelope and health checks are identical. azmlinfsrv calls
``init()`` once at startup and ``run(raw)`` per request.

Routing is by registered-model NAME + ``@champion`` alias load only
(``t_<slug>__m_<module>``). NO tag/alias *search* (unsupported on Azure ML). The
``tenant`` tag is checked AFTER load as defense-in-depth, never used to query.
"""

from __future__ import annotations

import json

# Imported from the pinned env (same wheel locally and in ACR).
from maxxflow_mlops.serving import ModelRouter

_router: ModelRouter | None = None


def init() -> None:
    global _router
    _router = ModelRouter()


def run(raw_data):
    """Request envelope:  {"tenant","module","records":[{...}]}  ->  predictions."""
    assert _router is not None, "init() was not called"
    payload = json.loads(raw_data) if isinstance(raw_data, (str, bytes, bytearray)) else raw_data
    tenant = payload["tenant"]
    module = payload["module"]
    records = payload.get("records", [])
    result = _router.predict(tenant=tenant, module=module, records=records)
    return result
