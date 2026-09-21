"""Ports for the ports/adapters seams (plan §1, §2, §10, §12a).

These Protocols are the *only* things module code depends on. Each has a LOCAL
adapter now (Postgres, MinIO/file, self-hosted MLflow, stub providers, in-proc
queue) and a documented seam for the Azure adapter later (read replica, ADLS,
Azure ML registry, Azure OpenAI, Service Bus / Event Grid). The adapter is
chosen by a factory that reads ``Settings`` — there is no ``if env==`` anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class DataSource(Protocol):
    """Reads tenant Postgres into DataFrames. Isolation is SET search_path."""

    def query(self, sql: str, params: Mapping[str, Any] | None = None,
              *, tenant: str | None = None) -> pd.DataFrame: ...

    def execute(self, sql: str, params: Mapping[str, Any] | None = None,
                *, tenant: str | None = None) -> None: ...


@runtime_checkable
class LakeIO(Protocol):
    """Medallion (bronze/silver/gold) read/write on fsspec. minio==adls; URI differs."""

    def write_parquet(self, df: pd.DataFrame, layer: str, name: str,
                      *, tenant: str, module: str) -> str: ...

    def read_parquet(self, layer: str, name: str,
                     *, tenant: str, module: str) -> pd.DataFrame: ...


@runtime_checkable
class ModelRegistry(Protocol):
    """MLflow tracking + registry. Route by NAME + ``@alias`` load ONLY.

    No tag/alias *search*, no stage-based deploy, no rename — those silently fail
    on Azure ML's MLflow registry (plan §2, §12a #1).
    """

    def log_and_register(self, model: Any, *, name: str, params: Mapping[str, Any],
                         metrics: Mapping[str, float], tags: Mapping[str, str],
                         signature: Any = None, input_example: Any = None) -> str: ...

    def load_champion(self, *, name: str, alias: str = "champion") -> Any: ...

    def set_alias(self, *, name: str, alias: str, version: str) -> None: ...

    def get_alias_version(self, *, name: str, alias: str) -> str | None: ...


@dataclass(frozen=True)
class GenerationConfig:
    """Generation controls a caller may request of an :class:`LLMProvider`.

    Not every field is honoured by every adapter — ``StubLLMProvider`` is
    already fully deterministic and ignores this; the Phase-2 Azure adapter
    is the one expected to actually vary its call based on it. Even at
    temperature 0 with a fixed seed, a real provider is not guaranteed to
    return byte-identical text run to run (batching/hardware nondeterminism
    is a known property of hosted LLM inference) — this narrows variance, it
    does not promise it away. Callers that need a true determinism guarantee
    get it from validating/normalising the response deterministically
    afterwards, not from this alone.
    """

    temperature: float = 0.0
    seed: int | None = None


@runtime_checkable
class LLMProvider(Protocol):
    """External LLM API (Azure OpenAI / OpenAI / Foundry). Text generation ONLY,
    NEVER scoring. Deterministic stub for local/CI."""

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        generation_config: GenerationConfig | None = None,
    ) -> str: ...

    @property
    def name(self) -> str: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """External embeddings API. OFF by default (TF-IDF is M4's default path).
    No transformer is hosted either way."""

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...

    @property
    def enabled(self) -> bool: ...

    @property
    def name(self) -> str: ...


@runtime_checkable
class EventQueue(Protocol):
    """M3/M4 retrain triggers. Local in-proc queue / APScheduler stub <-> Event
    Grid / Service Bus. The authoritative label/payload is carried IN the event
    envelope to beat replica lag (plan §6, §12a #3)."""

    def publish(self, topic: str, payload: Mapping[str, Any]) -> None: ...

    def drain(self, topic: str) -> list[dict]: ...


@runtime_checkable
class DriftReporter(Protocol):
    """Evidently locally (HTML) <-> Azure Monitor + Evidently in prod."""

    def report(self, reference: pd.DataFrame, current: pd.DataFrame,
               *, name: str, out_dir: str) -> dict: ...
