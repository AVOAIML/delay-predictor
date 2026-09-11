"""Persist completed Configurator training results in the tenant feature lake.

Layout: {LAKE_URI}/{tenant}/configurator/training-results/{model}/{run_id}.json
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import fsspec

from maxxflow_core.settings import Settings, get_settings

_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9_.-]+\Z")


def _segment(value: str, label: str) -> str:
    text = str(value)
    if not _SAFE_SEGMENT.fullmatch(text):
        raise ValueError(f"invalid {label}")
    return text


class TrainingResultStore:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def _uri(self, tenant: str, model_key: str, run_id: str) -> str:
        return (
            f"{self.settings.lake_uri.rstrip('/')}/{_segment(tenant, 'tenant')}/"
            f"configurator/training-results/{_segment(model_key, 'model key')}/"
            f"{_segment(run_id, 'run id')}.json"
        )

    def _filesystem(self, uri: str):
        opts = self.settings.lake_storage_options or None
        return fsspec.core.url_to_fs(uri, **(opts or {}))

    def save(self, snapshot: Mapping[str, Any]) -> str:
        tenant = str(snapshot["tenant"])
        model_key = str(snapshot["model_key"])
        run_id = str(snapshot["run_id"])
        uri = self._uri(tenant, model_key, run_id)
        fs, path = self._filesystem(uri)
        fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
        document = _json_safe({**dict(snapshot), "published": bool(snapshot.get("published"))})
        with fs.open(path, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        return uri

    def get(self, tenant: str, model_key: str, run_id: str) -> dict | None:
        uri = self._uri(tenant, model_key, run_id)
        fs, path = self._filesystem(uri)
        if not fs.exists(path):
            return None
        with fs.open(path, "r", encoding="utf-8") as stream:
            document = json.load(stream)
        if (document.get("tenant") != tenant or document.get("model_key") != model_key
                or document.get("run_id") != run_id):
            raise ValueError("stored training result identity does not match its lake path")
        return document

    def pending(self, tenant: str) -> list[dict]:
        root = (
            f"{self.settings.lake_uri.rstrip('/')}/{_segment(tenant, 'tenant')}/"
            "configurator/training-results/*/*.json"
        )
        fs, pattern = self._filesystem(root)
        documents = []
        for path in fs.glob(pattern):
            try:
                with fs.open(path, "r", encoding="utf-8") as stream:
                    item = json.load(stream)
                if (item.get("tenant") == tenant and item.get("status") == "done"
                        and not item.get("published", False)):
                    documents.append(item)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        documents.sort(key=lambda item: float(item.get("started_at") or 0), reverse=True)
        return documents

    def mark_published(self, tenant: str, model_key: str, version: str) -> None:
        pending = self.pending(tenant)
        published_at = next((
            float(item.get("started_at") or 0)
            for item in pending
            if item.get("model_key") == model_key
            and str((item.get("result") or {}).get("version")) == str(version)
        ), None)
        if published_at is None:
            return
        for item in pending:
            if (item.get("model_key") == model_key
                    and float(item.get("started_at") or 0) <= published_at):
                item["published"] = True
                self.save(item)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items() if key != "_model"}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    return str(value)


def get_training_result_store() -> TrainingResultStore:
    return TrainingResultStore()
