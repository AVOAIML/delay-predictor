"""Resolves a tenant's free-text description for the Weight Agent's cold-start
path (``tenant_description: str | None``), from one of three sources, in
priority order: request header -> database -> local YAML fallback.

The Weight Agent never learns which source a description came from — that
provenance is orchestrator-only telemetry (see ``resolve_description``'s
``TenantDescriptionResult``), and the description itself remains untrusted
free text all the way through: this reader only locates it, it does not
validate or interpret it. The existing profile-extraction trust boundary in
``llm_agents.weight_agent.profile_extractor`` is unchanged by this module.

Security invariant: every function here is keyed by an already-trusted
``tenant_id`` supplied by the caller. Nothing in this module ever derives,
infers, or overrides a tenant identity from a header, a database row, or the
local YAML file's own keys.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import yaml

from maxxflow_core.errors import MaXXFlowError, get_logger

from m3_production_delay.llm_agents.weight_agent.config import get_weight_agent_config
from m3_production_delay.llm_agents.weight_agent.tracing import NOOP_TRACER, WeightAgentTracer

log = get_logger("m3_production_delay.tenant_context_reader")

_TENANT_DESCRIPTION_HEADER = "x-tenant-description"
_DEFAULT_LOCAL_YAML_PATH = Path(__file__).resolve().parent / "meta_data" / "tenet_data.yml"

TenantDescriptionSource = Literal["request_header", "database", "local_yaml", "unavailable"]
SOURCE_REQUEST_HEADER: TenantDescriptionSource = "request_header"
SOURCE_DATABASE: TenantDescriptionSource = "database"
SOURCE_LOCAL_YAML: TenantDescriptionSource = "local_yaml"
SOURCE_UNAVAILABLE: TenantDescriptionSource = "unavailable"


class TenantContextError(MaXXFlowError):
    """Raised when the local YAML fallback file exists but cannot be trusted
    (parse failure, or the wrong top-level shape) — a deployment/config bug,
    not a normal "no description available" outcome, so it is not silently
    folded into ``None`` at the point it's detected. ``resolve_description``
    catches it and degrades to the next source / unavailable, exactly as it
    would treat any other source coming up empty."""


class TenantMetadataSource(Protocol):
    """DI seam for a future tenant-metadata table. No such table exists
    anywhere in this schema yet — verified against ``db/`` and
    ``maxxflow_data`` before writing this module. Implement this once one is
    approved; nothing here should be changed to wire it in."""

    def get_description(self, tenant_id: str) -> str | None: ...


class NullTenantMetadataSource:
    """Ships as the default. Mirrors
    ``llm_agents.weight_agent.providers.NullFittedWeightsProvider`` exactly:
    returns ``None`` unconditionally rather than querying a table that does
    not exist, which would either crash confusingly or require inventing a
    schema this module has no authority to define."""

    def get_description(self, tenant_id: str) -> str | None:
        return None


@dataclass(frozen=True)
class TenantDescriptionResult:
    description: str | None
    source: TenantDescriptionSource


class TenantContextReader:
    """Reads a tenant description from request headers, the tenant database,
    or a local YAML fallback, and exposes one priority-ordered
    ``resolve_description`` so callers never duplicate the source order
    themselves. Holds no per-tenant state — every method takes the tenant it
    concerns as an explicit argument.
    """

    def __init__(
        self,
        *,
        metadata_source: TenantMetadataSource | None = None,
        max_description_chars: int | None = None,
        local_yaml_path: Path | None = None,
    ) -> None:
        self._metadata_source = metadata_source or NullTenantMetadataSource()
        self._max_description_chars = (
            max_description_chars
            if max_description_chars is not None
            else get_weight_agent_config().max_tenant_description_chars
        )
        self._local_yaml_path = local_yaml_path or _DEFAULT_LOCAL_YAML_PATH

    def read_from_request_header(self, headers: Mapping[str, str]) -> str | None:
        """Descriptive context only — never a source of tenant identity. HTTP
        header names are case-insensitive, so the lookup is too; the caller
        (the orchestrator) is responsible for tenant_id coming from its own
        trusted context, not from anything in ``headers``."""
        for name, value in headers.items():
            if name.lower() != _TENANT_DESCRIPTION_HEADER:
                continue
            trimmed = value.strip()
            if not trimmed:
                return None
            if len(trimmed) > self._max_description_chars:
                log.info(
                    "tenant description header exceeded %d chars; truncated",
                    self._max_description_chars,
                )
                trimmed = trimmed[: self._max_description_chars]
            return trimmed
        return None

    def read_from_db(self, tenant_id: str) -> str | None:
        """Tenant-scoped by construction: ``tenant_id`` is passed straight
        through to the injected source, never used to build SQL here. A
        connectivity/runtime failure is treated the same as "no description
        available" (matches ``WeightAgent``'s own handling of
        ``FittedWeightsProvider`` failures) — it must not crash the rest of
        the resolution chain."""
        try:
            return self._metadata_source.get_description(tenant_id)
        except Exception as exc:  # provider transport/runtime failure
            log.warning("tenant %s metadata database read failed: %s", tenant_id, exc)
            return None

    def read_from_local(self, tenant_id: str) -> str | None:
        """Reads ``meta_data/tenet_data.yml`` next to this module (never the
        current working directory). Missing file, unknown tenant, and an
        empty description are all normal, controlled ``None`` outcomes.
        Malformed YAML is not — it raises :class:`TenantContextError` rather
        than being silently treated as "no data", since a broken bootstrap
        file is a deployment bug worth surfacing loudly.
        """
        if not self._local_yaml_path.is_file():
            log.info("local tenant metadata file not found at %s", self._local_yaml_path)
            return None
        try:
            raw_text = self._local_yaml_path.read_text(encoding="utf-8")
        except OSError as exc:
            log.error("could not read local tenant metadata file %s: %s", self._local_yaml_path, exc)
            return None

        try:
            payload = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            raise TenantContextError(f"local tenant metadata YAML is malformed: {exc}") from exc

        if payload is None:
            return None
        tenants = payload.get("tenants") if isinstance(payload, dict) else None
        if not isinstance(tenants, dict):
            raise TenantContextError(
                "local tenant metadata YAML must have a top-level 'tenants' mapping"
            )

        entry = tenants.get(tenant_id)
        if not isinstance(entry, dict):
            return None  # unknown tenant — never falls back to another tenant's entry
        description = entry.get("description")
        if not isinstance(description, str):
            return None
        return description.strip() or None

    def resolve_description(
        self,
        tenant_id: str,
        *,
        headers: Mapping[str, str] | None = None,
        tracer: WeightAgentTracer = NOOP_TRACER,
    ) -> TenantDescriptionResult:
        """First valid source wins — never merges descriptions across
        sources. Priority: request header (explicit runtime context for this
        request) -> database (persisted tenant-specific metadata) -> local
        YAML (development/bootstrap fallback)."""
        if headers is not None:
            description = self.read_from_request_header(headers)
            tracer.trace_stage(
                "TENANT CONTEXT", checking=SOURCE_REQUEST_HEADER, result="found" if description else "missing"
            )
            if description is not None:
                return self._log_result(tenant_id, description, SOURCE_REQUEST_HEADER, tracer)
        else:
            tracer.trace_stage("TENANT CONTEXT", checking=SOURCE_REQUEST_HEADER, result="not_supplied")

        description = self.read_from_db(tenant_id)
        tracer.trace_stage(
            "TENANT CONTEXT", checking=SOURCE_DATABASE, result="found" if description else "missing"
        )
        if description is not None:
            return self._log_result(tenant_id, description, SOURCE_DATABASE, tracer)

        try:
            description = self.read_from_local(tenant_id)
        except TenantContextError as exc:
            log.error("tenant %s local metadata fallback unusable: %s", tenant_id, exc)
            tracer.trace_decision("TENANT CONTEXT", "local_yaml_unusable", checking=SOURCE_LOCAL_YAML)
            description = None
        else:
            tracer.trace_stage(
                "TENANT CONTEXT", checking=SOURCE_LOCAL_YAML, result="found" if description else "missing"
            )
        if description is not None:
            return self._log_result(tenant_id, description, SOURCE_LOCAL_YAML, tracer)

        return self._log_result(tenant_id, None, SOURCE_UNAVAILABLE, tracer)

    @staticmethod
    def _log_result(
        tenant_id: str,
        description: str | None,
        source: TenantDescriptionSource,
        tracer: WeightAgentTracer = NOOP_TRACER,
    ) -> TenantDescriptionResult:
        log.info(
            "tenant_context_reader tenant_id=%s tenant_description_source=%s "
            "tenant_description_available=%s",
            tenant_id,
            source,
            description is not None,
        )
        # Content-gated: the raw description is only ever traced when the
        # caller explicitly turned trace-content on. Otherwise only its
        # length/source/availability are shown — never the text itself.
        if tracer.include_content:
            tracer.trace_stage(
                "TENANT DESCRIPTION RESOLVED",
                source=source,
                description_available=description is not None,
                description=description,
            )
        else:
            tracer.trace_stage(
                "TENANT DESCRIPTION RESOLVED",
                source=source,
                description_available=description is not None,
                description_length=len(description) if description else 0,
            )
        return TenantDescriptionResult(description=description, source=source)
