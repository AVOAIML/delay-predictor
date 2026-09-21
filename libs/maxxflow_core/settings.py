"""Environment-swapped configuration (plan §1, §2 "Secrets", §12a).

ONE codebase, environment-selected backends. The profile is chosen by the
``APP_ENV`` variable which maps (via a dict, never an ``if env==`` branch) to a
profile ``.env`` file under ``config/profiles/``. Real environment variables and
``.env.local`` override the profile defaults.

Module code NEVER compares the environment. It reads typed fields off
``get_settings()`` and asks the adapter factories (``maxxflow_core.ports`` +
each lib's ``factory``) for the backend. Swapping local→Azure is: change the
profile file / env vars and drop in the Azure adapter — no module edits.

The CI parity guard ``tests/parity/test_no_env_branches.py`` greps module source
for ``if env ==`` / ``APP_ENV ==`` and fails the build if any appears outside
this file.
"""

from __future__ import annotations

import functools
import os
import re
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = two parents up from this file (libs/maxxflow_core/settings.py)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROFILE_DIR = _REPO_ROOT / "config" / "profiles"

# APP_ENV -> profile file. A dict, deliberately, so there is no env-comparison
# branch anywhere in the codebase (the parity grep would flag one).
_PROFILE_FILES = {
    "local": _PROFILE_DIR / "local.env",
    # The deployed Azure profile. It was missing from this dict while
    # config/profiles/azure.env sat on disk, so APP_ENV=azure resolved to
    # nothing and fell through to LOCAL — a cloud container quietly pointing at
    # localhost. Nothing logged it. Adding a profile file is not enough; it has
    # to be registered here.
    "azure": _PROFILE_DIR / "azure.env",
    # PHASE-2 TEMPLATE, not deployable. Full of <...> markers by design.
    "dev-azure": _PROFILE_DIR / "dev-azure.env",
    "staging": _PROFILE_DIR / "staging.env",
    "prod": _PROFILE_DIR / "prod.env",
}
_DEFAULT_ENV = "local"

# `<user>`, `<replica-fqdn>`, `<account>` — the TODO markers the template
# profiles are made of. Deliberately not matched against '<' alone, so a
# password containing an angle bracket does not trip it.
_TEMPLATE_MARKER = re.compile(r"<[A-Za-z][\w.-]*>")


def _resolve_env_files() -> list[Path]:
    """Profile file first, then ``.env.local`` (local secrets), highest-priority last.

    An unrecognised APP_ENV is an ERROR, not a fallback. The previous
    ``.get(app_env, local)`` meant a typo or an unregistered profile name
    downgraded a cloud deployment to the local profile — localhost Postgres,
    localhost MinIO, localhost MLflow — and the only symptom was a connection
    error naming a host nobody configured. Failing at import names the actual
    problem.
    """
    app_env = os.environ.get("APP_ENV", _DEFAULT_ENV)
    if app_env not in _PROFILE_FILES:
        raise ValueError(
            f"APP_ENV={app_env!r} matches no profile. Known: "
            f"{', '.join(sorted(_PROFILE_FILES))}. Add the profile to "
            "_PROFILE_FILES in settings.py — a file in config/profiles/ that is "
            "not registered here can never be selected."
        )
    profile = _PROFILE_FILES[app_env]
    if not profile.exists():
        # Registered but absent is its own trap: settings fall back to bare
        # defaults plus whatever real env vars happen to be set, and the app
        # starts looking configured. azure.env was gitignored and therefore
        # missing from the container image, so this was one `git add` away from
        # happening for real.
        raise FileNotFoundError(
            f"APP_ENV={app_env!r} selects {profile}, which does not exist. If it is "
            "gitignored it is also absent from the container image. Ship the profile "
            "or pick a different APP_ENV — starting on bare defaults is not a "
            "recoverable state to debug."
        )
    files = [profile, _REPO_ROOT / ".env.local"]
    return [f for f in files if f.exists()]


class Settings(BaseSettings):
    """Typed configuration. Every backend URI/selector is a field here."""

    model_config = SettingsConfigDict(
        env_file=_resolve_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- environment identity -------------------------------------------------
    app_env: str = Field(default=_DEFAULT_ENV, alias="APP_ENV")

    # --- feature data source (Postgres: local container OR Azure read replica)
    # The DAL is the ONLY consumer. Tenant isolation is SET search_path, never
    # WHERE tenant_id (schema-per-tenant). Empty => DB-less mode (CI/unit tests).
    data_db_url: str = Field(default="", alias="DATA_DB_URL")
    # ...or assembled from parts when the password is a secret reference. See
    # _assemble_data_db_url below. An explicit DATA_DB_URL always wins.
    pg_host: str = Field(default="", alias="PGHOST")
    pg_port: int = Field(default=5432, alias="PGPORT")
    pg_user: str = Field(default="", alias="PGUSER")
    pg_password: SecretStr = Field(default=SecretStr(""), alias="PGPASS")
    pg_database: str = Field(default="", alias="PGDATABASE")
    pg_sslmode: str = Field(default="require", alias="PGSSLMODE")
    # ML reads features from the REPLICA in prod, never the primary (plan §1a).
    data_db_is_replica: bool = Field(default=False, alias="DATA_DB_IS_REPLICA")
    # One SQLAlchemy pool is shared by the process. Keep the limits explicit so a
    # Container App replica cannot open an unbounded number of Postgres sessions.
    db_pool_size: int = Field(default=5, alias="DB_POOL_SIZE", ge=1)
    db_pool_max_overflow: int = Field(default=10, alias="DB_POOL_MAX_OVERFLOW", ge=0)
    db_pool_timeout_seconds: int = Field(default=30, alias="DB_POOL_TIMEOUT_SECONDS", ge=1)
    db_pool_recycle_seconds: int = Field(default=300, alias="DB_POOL_RECYCLE_SECONDS", ge=1)

    # --- medallion lake (minio S3 local  <->  ADLS Gen2 prod) -----------------
    # Only the protocol differs (s3:// vs abfss://); set by env.
    lake_uri: str = Field(default="file://./lake", alias="LAKE_URI")
    lake_key: SecretStr = Field(default=SecretStr(""), alias="LAKE_KEY")
    lake_secret: SecretStr = Field(default=SecretStr(""), alias="LAKE_SECRET")
    lake_endpoint_url: str = Field(default="", alias="LAKE_ENDPOINT_URL")
    azure_storage_connection_string: SecretStr = Field(
        default=SecretStr(""), alias="AZURE_STORAGE_CONNECTION_STRING"
    )

    # --- MLflow (self-hosted local  <->  Azure ML workspace; same MLflow API) -
    mlflow_tracking_uri: str = Field(default="", alias="MLFLOW_TRACKING_URI")
    # Checked BEFORE mlflow_tracking_uri. Azure ML injects its own
    # MLFLOW_TRACKING_URI into every job container and it overrides whatever the
    # job spec set, which pointed training at the workspace's alias-less registry.
    # Nothing on the platform sets a MAXXFLOW_-prefixed name, so this survives.
    maxxflow_mlflow_uri: str = Field(default="", alias="MAXXFLOW_MLFLOW_URI")
    mlflow_registry_uri: str = Field(default="", alias="MLFLOW_REGISTRY_URI")
    # Route by registered-model NAME + this alias. NEVER tag/alias *search*
    # (unsupported on Azure ML's MLflow registry — plan §2, §12a #1).
    model_alias: str = Field(default="champion", alias="MODEL_ALIAS")

    # --- providers (external APIs behind ports; stub for local/CI) ------------
    # Classical models only on our infra. LLM/embeddings are API calls.
    llm_provider: str = Field(default="stub", alias="LLM_PROVIDER")
    embedding_provider: str = Field(default="stub", alias="EMBEDDING_PROVIDER")
    # M4 optional semantic boost — OFF by default (TF-IDF is the default path).
    embeddings_enabled: bool = Field(default=False, alias="EMBEDDINGS_ENABLED")
    # verify-before-building: confirm the deployed model id/region before use.
    llm_model_id: str = Field(default="", alias="LLM_MODEL_ID")
    embedding_model_id: str = Field(default="", alias="EMBEDDING_MODEL_ID")
    # Real LLM credentials (AzureOpenAILLMProvider, gated behind LLM_PROVIDER
    # in {"openai","azure_openai"} — the default "stub" reads none of these).
    openai_api_key: SecretStr = Field(default=SecretStr(""), alias="OPENAI_API_KEY")
    azure_openai_api_key: SecretStr = Field(default=SecretStr(""), alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field(default="", alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: str = Field(default="", alias="AZURE_OPENAI_API_VERSION")

    # --- secrets / determinism ------------------------------------------------
    # Operator UUIDs are PII; HMAC(salt)->tier in the DAL before bronze. The salt
    # is pinned (local secrets backend) so tiers are reproducible local==prod.
    hmac_salt: SecretStr = Field(default=SecretStr("local-dev-salt-change-me"), alias="HMAC_SALT")

    # --- tenancy --------------------------------------------------------------
    default_tenant: str = Field(default="demo", alias="DEFAULT_TENANT")
    # public schema holds tenant metadata; business data is per-tenant schema.
    tenant_schema_prefix: str = Field(default="tenant_", alias="TENANT_SCHEMA_PREFIX")

    # --- clock ----------------------------------------------------------------
    # Storage is UTC; presentation/business-day math uses this tz. Timestamps in
    # the schema are Timestamp(6) = TZ-naive, so we localise at the DAL boundary.
    presentation_tz: str = Field(default="Australia/Sydney", alias="PRESENTATION_TZ")
    # Frozen as-of clock for reproducible runs/tests (ISO-8601). Empty => now().
    as_of_override: str = Field(default="", alias="AS_OF_OVERRIDE")

    # --- where training runs --------------------------------------------------
    # "thread" = in this API process (works offline, no Azure); "azureml" = an
    # AML command job on its own compute. Selected here, never by an `if env ==`
    # anywhere else. See services/configurator/training_backends.py.
    train_backend: str = Field(default="thread", alias="TRAIN_BACKEND")
    aml_subscription_id: str = Field(default="", alias="AML_SUBSCRIPTION_ID")
    aml_resource_group: str = Field(default="", alias="AML_RESOURCE_GROUP")
    aml_workspace: str = Field(default="", alias="AML_WORKSPACE")
    # F-series by default: quota is per VM family, and the inference endpoint sits
    # on DSv2. Sharing a family means a training job queues behind the endpoint
    # that serves the model it is training.
    aml_instance_type: str = Field(default="Standard_F4s_v2", alias="AML_INSTANCE_TYPE")
    # The `train` stage of the one image family — same pinned env as the API.
    aml_train_image: str = Field(default="", alias="AML_TRAIN_IMAGE")
    # Client id of the USER-ASSIGNED identity an AML job runs as.
    #
    # Serverless compute has NO managed identity unless the job asks for one: it
    # runs with an AML token, which is scoped to the workspace and cannot read
    # Key Vault. DefaultAzureCredential then walks its whole chain and fails on
    # IMDS with "Expecting value: line 1 column 1" — a JSON parse error, which
    # reads like a bug rather than like "there is no identity here".
    #
    # Empty is the correct default: a CSV job needs no Azure resource beyond the
    # workspace, so it should not be handed an identity it does not use.
    aml_job_identity_client_id: str = Field(default="", alias="AML_JOB_IDENTITY_CLIENT_ID")
    # Where an AML job finds the Postgres password. The job is told the LOCATION,
    # never the value: its environment_variables are stored with the job and are
    # readable in Studio, so a password placed there persists in run history and
    # survives rotation. See libs/maxxflow_core/keyvault.py.
    keyvault_name: str = Field(default="", alias="KEYVAULT_NAME")
    pg_password_secret: str = Field(default="pg-admin-password", alias="PG_PASSWORD_SECRET")

    # --- serving --------------------------------------------------------------
    # BYOC multi-model router under azmlinfsrv. LRU size bounds resident models.
    model_lru_size: int = Field(default=8, alias="MODEL_LRU_SIZE")

    # --- M3 weight agent --------------------------------------------------------
    # Cold-start LLM-adjusted-prior path (modules/m3_production_delay/llm_agents/
    # weight_agent). Off => every non-configured, non-fitted resolution returns
    # the redistributed prior directly, same as an LLM failure would.
    m3_weight_llm_enabled: bool = Field(default=True, alias="M3_WEIGHT_AGENT_LLM_ENABLED")
    # Local/debug execution trace (modules/m3_production_delay/llm_agents/
    # weight_agent/tracing.py). Off in production by default; never read via
    # os.environ directly inside the Weight Agent or orchestrator — only
    # through these two settings. include_content additionally gates whether
    # raw tenant description / raw LLM prompt / raw LLM response text is
    # ever emitted, even when the trace itself is on.
    m3_weight_trace_enabled: bool = Field(default=False, alias="M3_WEIGHT_AGENT_TRACE_ENABLED")
    m3_weight_trace_include_content: bool = Field(
        default=False, alias="M3_WEIGHT_AGENT_TRACE_INCLUDE_CONTENT"
    )

    def tenant_schema(self, tenant_slug: str | None = None) -> str:
        slug = tenant_slug or self.default_tenant
        return f"{self.tenant_schema_prefix}{slug}"

    @property
    def lake_storage_options(self) -> dict:
        """Credentials/options for the fsspec implementation selected by LAKE_URI."""
        if self.lake_uri.startswith("s3://"):
            opts: dict = {
                "key": self.lake_key.get_secret_value() or None,
                "secret": self.lake_secret.get_secret_value() or None,
            }
            if self.lake_endpoint_url:
                opts["client_kwargs"] = {"endpoint_url": self.lake_endpoint_url}
            return {k: v for k, v in opts.items() if v is not None}
        if self.lake_uri.startswith(("abfs://", "abfss://")):
            connection_string = self.azure_storage_connection_string.get_secret_value()
            return {"connection_string": connection_string} if connection_string else {}
        return {}

    @model_validator(mode="after")
    def _assemble_data_db_url(self) -> "Settings":
        """Build DATA_DB_URL from parts when the password is a secret reference.

        THE PROBLEM. The canonical URL is one string with the password in the
        middle::

            postgresql+psycopg://mfadmin:${PGPASS}@pg-...:5432/configurator?sslmode=require

        Container Apps has no way to build that. A secret is injected as its OWN
        environment variable (``secretRef``); nothing interpolates ``${PGPASS}``
        into the middle of another variable's value, so setting DATA_DB_URL to
        the line above ships the six literal characters ``${PGPASS}`` as the
        password and every connection fails auth. Baking the password into the
        variable instead puts a Key Vault secret in plaintext on the app's
        template, where ``az containerapp show`` prints it.

        So: set PGHOST / PGUSER / PGDATABASE as plain variables and PGPASS as a
        secretRef, and the URL is assembled HERE, in process, from parts that
        were never concatenated anywhere a human or a CLI could read them.

        ON QUOTING. quote_plus on user and password is not decoration. Postgres
        passwords routinely contain ``@``, ``/``, ``:`` and ``#``; an unencoded
        ``@`` splits the userinfo early and SQLAlchemy then tries to resolve a
        hostname made of the rest of the password. The failure is a DNS error
        naming a fragment of the secret, which is both baffling and a leak. This
        matters more than it looks: rotating pg-admin-password is an open item,
        and the next rotation is exactly when a special character shows up.
        """
        if self.data_db_url:
            # An explicit URL wins — but catch the failure modes this method
            # exists to prevent, rather than letting them surface later as an
            # auth error or a DNS error.
            if "${" in self.data_db_url:
                raise ValueError(
                    "DATA_DB_URL contains an unexpanded ${...} placeholder. Nothing "
                    "interpolates it: Container Apps injects a secret as its own "
                    "variable, not into the middle of another one. Leave DATA_DB_URL "
                    "unset and provide PGHOST/PGUSER/PGDATABASE plus a PGPASS "
                    "secretRef instead.")
            # <...> markers come from the TEMPLATE profiles (dev-azure.env and
            # friends), which document the Phase-2 seam and are explicitly not
            # meant to be deployed. Unchecked, the first symptom is a DNS
            # failure on a hostname like '<replica-fqdn>' at the moment a user
            # clicks something — far from the config that caused it.
            if _TEMPLATE_MARKER.search(self.data_db_url):
                marker = _TEMPLATE_MARKER.search(self.data_db_url).group(0)
                raise ValueError(
                    f"DATA_DB_URL still contains the template marker {marker}. This "
                    "profile documents a seam rather than configuring one — check "
                    "APP_ENV. Leave DATA_DB_URL unset and provide "
                    "PGHOST/PGUSER/PGDATABASE plus a PGPASS secretRef instead.")
            return self
        if not (self.pg_host and self.pg_user and self.pg_database):
            return self          # DB-less mode — legitimate for CI and CSV-only runs
        # The same check, on the PARTS. Guarding only DATA_DB_URL left the obvious
        # hole open: infra/postgres-ha-experiment.sh documents repointing the app
        # at a replica with `--set-env-vars PGHOST=<replica fqdn>`, and a
        # placeholder pasted verbatim sails through to fail much later as
        # "failed to resolve host '<replica-fqdn>'" — surfacing when a user clicks
        # Connect Database, a deploy away from the config that caused it.
        for field, value in (("PGHOST", self.pg_host), ("PGUSER", self.pg_user),
                             ("PGDATABASE", self.pg_database)):
            hit = _TEMPLATE_MARKER.search(value)
            if hit:
                raise ValueError(
                    f"{field} is set to the placeholder {hit.group(0)} — a template "
                    "marker that got pasted rather than substituted. Set the real "
                    "value, or unset it for DB-less mode.")
            if "${" in value:
                raise ValueError(
                    f"{field} contains an unexpanded ${{...}} placeholder. Container "
                    "Apps does not interpolate one variable into another; set the "
                    "literal value.")
        from urllib.parse import quote_plus
        user = quote_plus(self.pg_user)
        password = quote_plus(self.pg_password.get_secret_value())
        auth = f"{user}:{password}" if password else user
        self.data_db_url = (f"postgresql+psycopg://{auth}@{self.pg_host}:{self.pg_port}"
                            f"/{self.pg_database}?sslmode={self.pg_sslmode}")
        return self

    @property
    def db_enabled(self) -> bool:
        return bool(self.data_db_url)

    @property
    def data_db_has_password(self) -> bool:
        """Does the effective URL carry a password?

        Not the same question as "is PGPASS set". An explicit DATA_DB_URL (docker
        compose, a developer's .env.local) embeds the password and leaves PGPASS
        empty, so checking the field alone would call a perfectly good local
        connection credential-less.
        """
        url = self.data_db_url
        if not url or "@" not in url:
            return False
        _, _, rest = url.partition("://")
        userinfo, _, _host = rest.rpartition("@")
        return ":" in userinfo

    @property
    def data_db_url_safe(self) -> str:
        """The URL with the password masked — the ONLY form that may be logged.

        ``data_db_url`` is a plain str, so it renders in full in a traceback, a
        repr, or a well-meaning debug log. Anything that wants to show a human
        which database it is talking to must use this."""
        url = self.data_db_url
        if not url or "@" not in url:
            return url
        scheme, _, rest = url.partition("://")
        userinfo, _, hostpart = rest.rpartition("@")
        user, sep, _secret = userinfo.partition(":")
        if not user:
            return url
        # Distinguish "masked" from "absent". Printing *** for an empty password
        # made a missing credential look like a present one: an AML job logged
        # `mfadmin:***@pg-...` and then failed with "no password supplied", which
        # reads as the server rejecting a password rather than as there being none.
        shown = "***" if sep else "(no password)"
        return f"{scheme}://{user}:{shown}@{hostpart}"


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton. Call ``get_settings.cache_clear()`` in tests to reload."""
    return Settings()
