"""AZURE_STORAGE_CONTAINER: ``local`` = MinIO, anything else = that Azure container.

Nothing here touches a network: building a LakeIO only records a root and its
options. The conftest guard blanks the real Azure storage credentials, so each
test passes exactly the values it means.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from maxxflow_core.settings import Settings
from maxxflow_features.lake import LakeIO
from m3_production_delay.snapshots.store import LAYER, MODULE, get_snapshot_lake, snapshot_name

_REPO = Path(__file__).resolve().parents[4]
ACCOUNT = "mfstorage"
CONNECTION_STRING = (
    f"DefaultEndpointsProtocol=https;AccountName={ACCOUNT};"
    "AccountKey=c2VjcmV0LWtleQ==;EndpointSuffix=core.windows.net"
)
AZURE_ROOT = f"abfss://dev@{ACCOUNT}.dfs.core.windows.net/maxxflow-lake"


def _local(**extra) -> Settings:
    return Settings(
        AZURE_STORAGE_CONTAINER="local",
        LAKE_URI="s3://maxxflow-lake",
        LAKE_KEY="minioadmin",
        LAKE_SECRET="minioadmin",
        LAKE_ENDPOINT_URL="http://localhost:9010",
        **extra,
    )


def test_the_code_default_is_the_azure_dev_container():
    assert Settings.model_fields["azure_storage_container"].default == "dev"


def test_the_local_profile_selects_minio():
    profile = (_REPO / "config" / "profiles" / "local.env").read_text(encoding="utf-8")
    assert "AZURE_STORAGE_CONTAINER=local" in profile.splitlines()


@pytest.mark.parametrize("value", ["local", "LOCAL", " Local "])
def test_local_uses_the_minio_lake(value):
    s = _local()
    s = s.model_copy(update={"azure_storage_container": value})
    assert s.storage_is_local
    assert s.container_lake_uri == "s3://maxxflow-lake"
    assert s.container_lake_options == {
        "key": "minioadmin",
        "secret": "minioadmin",
        "client_kwargs": {"endpoint_url": "http://localhost:9010"},
    }


def test_dev_targets_that_azure_container_with_the_connection_string():
    s = Settings(
        AZURE_STORAGE_CONTAINER="dev",
        AZURE_STORAGE_ACCOUNT=ACCOUNT,
        AZURE_STORAGE_CONNECTION_STRING=CONNECTION_STRING,
    )
    assert not s.storage_is_local
    assert s.container_lake_uri == AZURE_ROOT
    assert s.container_lake_options == {"connection_string": CONNECTION_STRING}


def test_the_account_is_read_from_the_connection_string_when_not_set():
    s = Settings(AZURE_STORAGE_CONTAINER="dev", AZURE_STORAGE_CONNECTION_STRING=CONNECTION_STRING)
    assert s.container_lake_uri == AZURE_ROOT


def test_without_a_connection_string_the_apps_own_identity_is_used():
    s = Settings(AZURE_STORAGE_CONTAINER="dev", AZURE_STORAGE_ACCOUNT=ACCOUNT)
    assert s.container_lake_options == {"account_name": ACCOUNT, "anon": False}


def test_azure_without_an_account_fails_naming_the_fix():
    s = Settings(AZURE_STORAGE_CONTAINER="dev")
    with pytest.raises(ValueError, match="AZURE_STORAGE_ACCOUNT") as excinfo:
        s.container_lake_uri
    assert "AZURE_STORAGE_CONTAINER=local" in str(excinfo.value)


@pytest.mark.parametrize("bad", ["local/dev", "Dev", "d", "dev_lake", "dev--lake", "-dev"])
def test_an_invalid_container_name_is_refused_up_front(bad):
    s = Settings(AZURE_STORAGE_CONTAINER=bad, AZURE_STORAGE_ACCOUNT=ACCOUNT)
    with pytest.raises(ValueError, match="valid Azure container name"):
        s.container_lake_uri


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [("", f"abfss://dev@{ACCOUNT}.dfs.core.windows.net"),
     ("/tenants/lake/", f"abfss://dev@{ACCOUNT}.dfs.core.windows.net/tenants/lake")],
)
def test_the_prefix_is_optional_and_normalised(prefix, expected):
    s = Settings(
        AZURE_STORAGE_CONTAINER="dev", AZURE_STORAGE_ACCOUNT=ACCOUNT, AZURE_STORAGE_PREFIX=prefix
    )
    assert s.container_lake_uri == expected


def test_below_the_root_minio_and_azure_get_the_same_path():
    name = snapshot_name(
        "WH/MO/00142", dt.datetime(2026, 9, 24, 11, 17, tzinfo=dt.timezone(dt.timedelta(hours=10)))
    )
    local = LakeIO.at("s3://maxxflow-lake")._uri(LAYER, name, tenant="demo", module=MODULE, ext="json")
    azure = LakeIO.at(AZURE_ROOT)._uri(LAYER, name, tenant="demo", module=MODULE, ext="json")
    assert local.removeprefix("s3://") == azure.removeprefix(
        f"abfss://dev@{ACCOUNT}.dfs.core.windows.net/"
    )


def test_the_snapshot_store_follows_the_setting():
    assert get_snapshot_lake(_local()).root == "s3://maxxflow-lake"
    azure = Settings(
        AZURE_STORAGE_CONTAINER="dev",
        AZURE_STORAGE_ACCOUNT=ACCOUNT,
        AZURE_STORAGE_CONNECTION_STRING=CONNECTION_STRING,
    )
    lake = get_snapshot_lake(azure)
    assert lake.root == AZURE_ROOT
    assert "AccountKey" not in lake.root  # the secret stays in options, never the URI


def test_a_lake_rooted_explicitly_leaves_lake_uri_alone():
    # M1/M2 keep writing where LAKE_URI points; only the snapshot store moves.
    s = _local()
    s = s.model_copy(update={"azure_storage_container": "dev", "azure_storage_account": ACCOUNT})
    assert s.lake_uri == "s3://maxxflow-lake"
    assert s.container_lake_uri == AZURE_ROOT
