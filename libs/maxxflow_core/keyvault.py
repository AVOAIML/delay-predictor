"""Fetch the Postgres password at runtime, for compute that has no secretRef.

WHY THIS EXISTS
---------------
ca-configurator gets PGPASS from a Container Apps secret reference: the platform
resolves it from Key Vault and injects it as an environment variable, and nothing
in the app template holds the value. An Azure ML job has no equivalent. Its
`environment_variables` are part of the job definition, stored with the job and
readable in Studio by anyone with workspace access — and kept in the history of
every run, so a rotation does not retract them.

So the job is told WHERE the secret is, not what it is, and fetches it itself
using the compute's managed identity. What the job definition carries is a vault
name and a secret name, both of which are safe to read.

DELIBERATELY A NO-OP WHEN NOT NEEDED
------------------------------------
If PGPASS is already set (Container Apps, docker compose, a developer's shell) or
no vault is configured (CI, a laptop, the test suite), this returns without
importing anything Azure. The import is lazy for the same reason the AML client's
is: azure-keyvault-secrets lives in an optional extra, and the local path must
keep working in an image that does not have it.
"""

from __future__ import annotations

import os

from maxxflow_core.errors import get_logger

log = get_logger("core.keyvault")

_VAULT_ENV = "KEYVAULT_NAME"
_SECRET_ENV = "PG_PASSWORD_SECRET"
_DEFAULT_SECRET = "pg-admin-password"


def hydrate_pg_password() -> bool:
    """Set PGPASS from Key Vault if it is missing and a vault is configured.

    Returns True if it fetched one. Must be called BEFORE get_settings(), which is
    lru_cached and reads the environment once — a later call would build the
    connection URL from an empty password and fail authentication instead.
    """
    if os.environ.get("PGPASS"):
        return False                      # already injected — nothing to do
    vault = os.environ.get(_VAULT_ENV, "").strip()
    if not vault:
        return False                      # no vault configured; DB-less or PGPASS-less by design

    secret_name = os.environ.get(_SECRET_ENV, "").strip() or _DEFAULT_SECRET
    try:
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient
    except ImportError as e:
        raise RuntimeError(
            f"{_VAULT_ENV}={vault} is set but azure-keyvault-secrets / azure-identity "
            "are not installed in this image. Rebuild with the `aml` extra, or supply "
            "PGPASS directly.") from e

    url = f"https://{vault}.vault.azure.net"
    try:
        client = SecretClient(vault_url=url, credential=DefaultAzureCredential())
        os.environ["PGPASS"] = client.get_secret(secret_name).value
    except Exception as e:
        # Name the identity problem explicitly. The failure is almost always an
        # access policy rather than a wrong name, and the raw Azure error says
        # "Forbidden" without saying whose forbidden it is.
        raise RuntimeError(
            f"could not read secret '{secret_name}' from {url} ({type(e).__name__}: {e}). "
            "The compute needs an identity WITH GET on secrets in that vault, and on "
            "Azure ML those are two separate problems.\n"
            "  1. A serverless job has no managed identity unless it declares one — "
            "system-assigned is not supported there, so it must be a USER-ASSIGNED "
            "identity attached to the workspace and named by "
            "AML_JOB_IDENTITY_CLIENT_ID. Without it the credential chain finds "
            "nothing on IMDS and reports a JSON parse error.\n"
            "  2. That identity then needs GET on secrets (an access policy, or the "
            "'Key Vault Secrets User' role on an RBAC vault).\n"
            "`./infra/aml/db-access.sh grant` does both, and prints the env var to "
            "set.") from e

    log.info("PGPASS loaded from key vault %s (secret %s)", vault, secret_name)
    return True
