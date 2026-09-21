from __future__ import annotations

from pathlib import Path

import pytest

from m3_production_delay.llm_agents.weight_agent.tracing import WeightAgentTracer
from m3_production_delay.tenant_context_reader import (
    SOURCE_DATABASE,
    SOURCE_LOCAL_YAML,
    SOURCE_REQUEST_HEADER,
    SOURCE_UNAVAILABLE,
    TenantContextError,
    TenantContextReader,
)

MAX_CHARS = 40  # small, deliberately, so the truncation test doesn't need a huge fixture


def _reader(**kwargs) -> TenantContextReader:
    kwargs.setdefault("max_description_chars", MAX_CHARS)
    return TenantContextReader(**kwargs)


class _FixedMetadataSource:
    def __init__(self, value: str | None = None, *, raises: Exception | None = None) -> None:
        self._value = value
        self._raises = raises
        self.calls: list[str] = []

    def get_description(self, tenant_id: str) -> str | None:
        self.calls.append(tenant_id)
        if self._raises is not None:
            raise self._raises
        return self._value


# --- read_from_request_header ------------------------------------------------


def test_header_present_returns_trimmed_description():
    reader = _reader()
    result = reader.read_from_request_header({"x-tenant-description": "  hello there  "})
    assert result == "hello there"


def test_header_missing_returns_none():
    reader = _reader()
    assert reader.read_from_request_header({"content-type": "application/json"}) is None


def test_header_empty_returns_none():
    reader = _reader()
    assert reader.read_from_request_header({"x-tenant-description": "   "}) is None


def test_header_too_long_is_truncated_to_configured_limit():
    reader = _reader()
    long_value = "x" * (MAX_CHARS + 50)
    result = reader.read_from_request_header({"x-tenant-description": long_value})
    assert result == "x" * MAX_CHARS


def test_header_lookup_is_case_insensitive():
    reader = _reader()
    result = reader.read_from_request_header({"X-Tenant-Description": "case test"})
    assert result == "case test"


def test_header_cannot_supply_or_change_tenant_identity():
    # The header source has no tenant_id parameter at all — it cannot read or
    # influence it even if a caller stuffs an id-like key into the headers.
    reader = _reader()
    result = reader.read_from_request_header(
        {"x-tenant-description": "real description", "x-tenant-id": "spoofed-tenant"}
    )
    assert result == "real description"
    resolved = reader.resolve_description(
        "trusted-tenant",
        headers={"x-tenant-description": "real description", "x-tenant-id": "spoofed-tenant"},
    )
    assert resolved.description == "real description"
    # tenant_id used for the rest of resolution is whatever the caller passed
    # in, never anything pulled from the headers dict.


# --- read_from_local ----------------------------------------------------------


def _write_yaml(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "tenet_data.yml"
    path.write_text(content, encoding="utf-8")
    return path


def test_local_known_tenant_returns_correct_description(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        tenants:
          tenant_001:
            description: "Tenant one description."
          tenant_002:
            description: "Tenant two description."
        """,
    )
    reader = _reader(local_yaml_path=path, max_description_chars=1000)
    assert reader.read_from_local("tenant_001") == "Tenant one description."


def test_local_never_returns_another_tenants_description(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        tenants:
          tenant_001:
            description: "Tenant one description."
          tenant_002:
            description: "Tenant two description."
        """,
    )
    reader = _reader(local_yaml_path=path, max_description_chars=1000)
    assert reader.read_from_local("tenant_002") == "Tenant two description."
    assert reader.read_from_local("tenant_002") != reader.read_from_local("tenant_001")


def test_local_unknown_tenant_returns_none(tmp_path):
    path = _write_yaml(tmp_path, "tenants:\n  tenant_001:\n    description: 'x'\n")
    reader = _reader(local_yaml_path=path)
    assert reader.read_from_local("does-not-exist") is None


def test_local_missing_file_returns_none(tmp_path):
    reader = _reader(local_yaml_path=tmp_path / "does-not-exist.yml")
    assert reader.read_from_local("tenant_001") is None


def test_local_empty_description_returns_none(tmp_path):
    path = _write_yaml(tmp_path, "tenants:\n  tenant_001:\n    description: '   '\n")
    reader = _reader(local_yaml_path=path)
    assert reader.read_from_local("tenant_001") is None


def test_local_malformed_yaml_raises_controlled_exception(tmp_path):
    path = _write_yaml(tmp_path, "tenants: [unterminated\n  - broken")
    reader = _reader(local_yaml_path=path)
    with pytest.raises(TenantContextError):
        reader.read_from_local("tenant_001")


def test_local_wrong_top_level_shape_raises_controlled_exception(tmp_path):
    path = _write_yaml(tmp_path, "not_tenants:\n  tenant_001:\n    description: 'x'\n")
    reader = _reader(local_yaml_path=path)
    with pytest.raises(TenantContextError):
        reader.read_from_local("tenant_001")


# --- read_from_db --------------------------------------------------------------


def test_db_description_found():
    source = _FixedMetadataSource("db-sourced description")
    reader = _reader(metadata_source=source)
    assert reader.read_from_db("tenant_001") == "db-sourced description"


def test_db_tenant_not_found_returns_none():
    source = _FixedMetadataSource(None)
    reader = _reader(metadata_source=source)
    assert reader.read_from_db("tenant_001") is None


def test_db_unavailable_is_caught_and_returns_none():
    source = _FixedMetadataSource(raises=ConnectionError("db down"))
    reader = _reader(metadata_source=source)
    assert reader.read_from_db("tenant_001") is None


def test_db_tenant_isolation_is_preserved():
    source = _FixedMetadataSource("only ever this value")
    reader = _reader(metadata_source=source)
    reader.read_from_db("tenant_a")
    reader.read_from_db("tenant_b")
    assert source.calls == ["tenant_a", "tenant_b"]


# --- source priority via resolve_description ----------------------------------


def test_priority_header_wins_over_db_and_local(tmp_path):
    path = _write_yaml(tmp_path, "tenants:\n  t:\n    description: 'local'\n")
    reader = _reader(metadata_source=_FixedMetadataSource("db"), local_yaml_path=path, max_description_chars=1000)
    result = reader.resolve_description("t", headers={"x-tenant-description": "header"})
    assert (result.description, result.source) == ("header", SOURCE_REQUEST_HEADER)


def test_priority_db_wins_over_local_when_no_header(tmp_path):
    path = _write_yaml(tmp_path, "tenants:\n  t:\n    description: 'local'\n")
    reader = _reader(metadata_source=_FixedMetadataSource("db"), local_yaml_path=path, max_description_chars=1000)
    result = reader.resolve_description("t")
    assert (result.description, result.source) == ("db", SOURCE_DATABASE)


def test_priority_local_wins_when_only_local_available(tmp_path):
    path = _write_yaml(tmp_path, "tenants:\n  t:\n    description: 'local'\n")
    reader = _reader(local_yaml_path=path, max_description_chars=1000)
    result = reader.resolve_description("t")
    assert (result.description, result.source) == ("local", SOURCE_LOCAL_YAML)


def test_priority_nothing_available_is_unavailable(tmp_path):
    reader = _reader(local_yaml_path=tmp_path / "missing.yml")
    result = reader.resolve_description("t", headers={})
    assert (result.description, result.source) == (None, SOURCE_UNAVAILABLE)


def test_priority_malformed_local_degrades_to_unavailable_instead_of_raising(tmp_path):
    path = _write_yaml(tmp_path, "tenants: [unterminated\n  - broken")
    reader = _reader(local_yaml_path=path)
    result = reader.resolve_description("t")
    assert (result.description, result.source) == (None, SOURCE_UNAVAILABLE)


# --- trace never leaks a secret-looking header ------------------------------


class _RecordingSink:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, msg: object, *args: object) -> None:
        self.messages.append(str(msg) % args if args else str(msg))


def test_authorization_like_headers_never_appear_in_the_trace(tmp_path):
    tracer = WeightAgentTracer(trace_id="t1", enabled=True, include_content=True)
    sink = _RecordingSink()
    tracer._log = sink  # type: ignore[assignment]
    reader = _reader(local_yaml_path=tmp_path / "missing.yml")

    reader.resolve_description(
        "t",
        headers={
            "authorization": "Bearer super-secret-token",
            "cookie": "session=abc123",
            "x-tenant-description": "a public tenant description",
        },
        tracer=tracer,
    )

    text = "\n".join(sink.messages)
    assert "super-secret-token" not in text
    assert "session=abc123" not in text
    assert "Bearer" not in text
    assert "a public tenant description" in text  # the legitimate field is still visible
