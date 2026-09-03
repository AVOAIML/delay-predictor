"""LakeIO — medallion read/write via fsspec/pyarrow (plan §2 lake row).

URI layout: ``{lake_uri}/{tenant}/{module}/{layer}/{name}.parquet``. The only
thing that changes local→Azure is the protocol (``s3://`` ⇄ ``abfss://``) and the
storage options, both from ``Settings`` — code is identical.
"""

from __future__ import annotations

import pandas as pd

from maxxflow_core.settings import Settings, get_settings


class LakeIO:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def _uri(self, layer: str, name: str, *, tenant: str, module: str) -> str:
        base = self.settings.lake_uri.rstrip("/")
        return f"{base}/{tenant}/{module}/{layer}/{name}.parquet"

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
