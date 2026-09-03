"""BYOC multi-model router (plan §7, §12a #1, #2).

ONE always-on endpoint serves many (tenant, module) models. The router is what
``docker/score.py`` calls under azmlinfsrv — identical local and on the AML
managed online endpoint. Cold-miss = LRU load of ``models:/name@champion``; the
``tenant`` tag is asserted AFTER load as defense-in-depth, never used to query.

Classical-only models (no torch in-process) keep the working set small, so a
modest LRU covers the active tenants.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import pandas as pd

from maxxflow_core.errors import TenantIsolationError
from maxxflow_core.settings import get_settings
from maxxflow_mlops.naming import GLOBAL_TENANT
from maxxflow_mlops.registry import MLflowRegistry


class ModelRouter:
    def __init__(self, registry: MLflowRegistry | None = None, lru_size: int | None = None):
        self.registry = registry or MLflowRegistry()
        self.lru_size = lru_size or get_settings().model_lru_size
        self._alias = get_settings().model_alias
        self._cache: "OrderedDict[str, Any]" = OrderedDict()

    def _get_or_load(self, name: str, tenant: str):
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        model = self.registry.load_champion(name=name, alias=self._alias)
        # Defense-in-depth: the registered NAME already isolates the tenant; the
        # tag assertion catches a mis-registered artifact. NOT a query (§12a #1).
        # The shared GLOBAL base model is legitimately served to any tenant, so its
        # `global` tag is allowed alongside the requesting tenant's own tag.
        tags = self.registry.get_alias_tags(name=name, alias=self._alias)
        if tags.get("tenant") not in (None, tenant, GLOBAL_TENANT):
            raise TenantIsolationError(
                f"model {name} carries tenant tag {tags.get('tenant')!r} != request {tenant!r}"
            )
        self._cache[name] = model
        self._cache.move_to_end(name)
        while len(self._cache) > self.lru_size:
            self._cache.popitem(last=False)
        return model

    def predict(self, *, tenant: str, module: str, records: list[dict]) -> dict:
        # Onboarding fallback: serve the tenant's own champion, else the GLOBAL base.
        resolved = self.registry.resolve_champion_name(
            tenant=tenant, module=module, alias=self._alias)
        if resolved is None:
            raise TenantIsolationError(
                f"no champion for tenant {tenant!r} module {module!r} and no global base "
                f"model to fall back to — train and publish one first")
        name, is_base = resolved
        model = self._get_or_load(name, tenant)
        version = self.registry.get_alias_version(name=name, alias=self._alias)
        frame = _coerce_to_signature(pd.DataFrame(records), model)
        preds = model.predict(frame)
        # Normalise to JSON-serialisable records.
        if isinstance(preds, pd.DataFrame):
            out = preds.to_dict(orient="records")
        elif hasattr(preds, "tolist"):
            out = preds.tolist()
        else:
            out = list(preds)
        return {"model_name": name, "model_version": version,
                "is_base_model": is_base, "predictions": out}


def _coerce_to_signature(frame: pd.DataFrame, model) -> pd.DataFrame:
    """Cast incoming columns to the dtypes the model's signature declares.

    JSON has one number type. A caller sending ``"quantity": 15`` produces an
    int64 column, MLflow's schema enforcement refuses to widen it, and the whole
    request fails with

        Incompatible input types for column quantity.
        Can not safely convert int64 to float64.

    which blames the data for something no JSON client can express: there is no
    way to write "the float fifteen" in a JSON body. Every whole-number quantity,
    lead time or price would hit this. Widening int -> float here is lossless and
    is the conversion MLflow itself declines to do implicitly.

    Deliberately one-directional. float -> int is NOT performed: that would
    silently truncate, and a request whose 15.7 became 15 is worse than a request
    that failed. Unknown columns are left alone — MLflow drops the ones its
    signature does not name, and warns, which is the behaviour we want.
    """
    try:
        schema = model.metadata.get_input_schema()
        if schema is None:
            return frame
        wanted = dict(zip(schema.input_names(), schema.input_types()))
    except Exception:                       # unsigned model: nothing to enforce
        return frame

    import numpy as np

    out = frame
    for name, dtype in wanted.items():
        if name not in out.columns:
            continue
        target = getattr(dtype, "to_numpy", lambda: None)()
        if target is None:
            continue
        current = out[name].dtype
        if current == target:
            continue
        # Only widen. int64 -> float64 is exact for the magnitudes in a quote;
        # anything else is left for MLflow to accept or reject on its own terms.
        if np.issubdtype(current, np.integer) and np.issubdtype(target, np.floating):
            if out is frame:
                out = frame.copy()
            out[name] = out[name].astype(target)
    return out
