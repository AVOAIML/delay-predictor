"""LakeIO — medallion read/write via fsspec/pyarrow (plan §2 lake row).

URI layout: ``{lake_uri}/{tenant}/{module}/{layer}/{name}.{parquet|json}``. The
only thing that changes local→Azure is the protocol (``s3://`` ⇄ ``abfss://``) and
the storage options, both from ``Settings`` — code is identical.

Parquet holds frames (features, the gold tier). JSON holds single records that
the bronze tier keeps verbatim — one event per object, e.g. M3's per-scoring
snapshots — where forcing a one-row frame would mean fixing a flat schema at
ingest, which is exactly what bronze exists to avoid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from maxxflow_core.settings import Settings, get_settings


@dataclass(frozen=True)
class _Root:
    """The two settings LakeIO reads, for a lake not rooted at LAKE_URI."""

    lake_uri: str
    lake_storage_options: dict = field(default_factory=dict)


class LakeIO:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    @classmethod
    def at(cls, uri: str, storage_options: dict | None = None) -> "LakeIO":
        """A lake rooted at an explicit URI instead of LAKE_URI — for a store
        another setting selects (e.g. ``Settings.container_lake_uri``). Same
        layout below the root, so the same paths land in either store."""
        return cls(_Root(uri, dict(storage_options or {})))  # type: ignore[arg-type]

    @property
    def root(self) -> str:
        """The lake's root URI. Never carries credentials — those live in options."""
        return self.settings.lake_uri.rstrip("/")

    def _uri(self, layer: str, name: str, *, tenant: str, module: str, ext: str = "parquet") -> str:
        base = self.settings.lake_uri.rstrip("/")
        return f"{base}/{tenant}/{module}/{layer}/{name}.{ext}"

    def _fs(self, uri: str):
        import fsspec
        opts = self.settings.lake_storage_options or None
        return fsspec.core.url_to_fs(uri, **(opts or {}))

    def write_json(self, obj: Any, layer: str, name: str, *, tenant: str, module: str) -> str:
        """Write one JSON document, replacing any object already at that name.

        ``obj`` must already be JSON-safe. Serialization is strict
        (``allow_nan=False``): a stray NaN/Infinity raises here rather than
        landing as a literal most JSON readers reject.
        """
        uri = self._uri(layer, name, tenant=tenant, module=module, ext="json")
        data = json.dumps(obj, allow_nan=False, ensure_ascii=False, sort_keys=True).encode("utf-8")
        fs, path = self._fs(uri)
        parent = path.rsplit("/", 1)[0]
        try:
            fs.makedirs(parent, exist_ok=True)
        except Exception:
            pass
        with fs.open(path, "wb") as fh:
            fh.write(data)
        return uri

    def read_json(self, layer: str, name: str, *, tenant: str, module: str) -> Any | None:
        """The document at that name, or None when nothing has been written there."""
        uri = self._uri(layer, name, tenant=tenant, module=module, ext="json")
        fs, path = self._fs(uri)
        if not fs.exists(path):
            return None
        with fs.open(path, "rb") as fh:
            return json.loads(fh.read().decode("utf-8"))

    def write_parquet(self, df: pd.DataFrame, layer: str, name: str, *, tenant: str, module: str) -> str:
        uri = self._uri(layer, name, tenant=tenant, module=module)
        opts = self.settings.lake_storage_options or None
        # ensure the parent "directory" exists (local fs needs it; S3 is a no-op)
        import fsspec
        fs, path = fsspec.core.url_to_fs(uri, **(opts or {}))
        parent = path.rsplit("/", 1)[0]
        try:
            fs.makedirs(parent, exist_ok=True)
        except Exception:
            pass
        _sanitize_for_parquet(df).to_parquet(uri, index=False, storage_options=opts)
        return uri

    def read_parquet(self, layer: str, name: str, *, tenant: str, module: str) -> pd.DataFrame:
        uri = self._uri(layer, name, tenant=tenant, module=module)
        return pd.read_parquet(uri, storage_options=self.settings.lake_storage_options or None)


def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce object columns Arrow can't infer: uuid.UUID -> str, Decimal -> float.
    The precision-critical Decimal math already happened in features; carry-columns
    (e.g. grand_total) are fine as float in the gold layer."""
    import decimal
    import uuid as _uuid
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(
                lambda v: str(v) if isinstance(v, _uuid.UUID)
                else float(v) if isinstance(v, decimal.Decimal) else v
            )
    return out


def get_lake() -> LakeIO:
    return LakeIO()
