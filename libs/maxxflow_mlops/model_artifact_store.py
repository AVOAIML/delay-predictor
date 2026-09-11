"""Durable pyfunc model artifacts stored directly in the tenant feature lake.

MLflow remains the version and alias registry, but serving must not depend on the
tracking server's local artifact filesystem. Every training run is stored under
``{LAKE_URI}/{tenant}/model-artifacts/{module}/runs/{run_id}/model`` and the URI
is attached to the exact MLflow model version created for that run.
"""

from __future__ import annotations

import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import fsspec
import mlflow

from maxxflow_core.settings import Settings, get_settings
from maxxflow_mlops.naming import parse_model_name

LAKE_ARTIFACT_URI_TAG = "maxxflow.lake_artifact_uri"


class ModelArtifactStore:
    """Save and load immutable MLflow pyfunc bundles through ``fsspec``."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def uri(self, name: str, run_id: str) -> str:
        tenant, module = parse_model_name(name)
        return (
            f"{self.settings.lake_uri.rstrip('/')}/{tenant}/model-artifacts/"
            f"{module}/runs/{run_id}/model"
        )

    def _filesystem(self, uri: str):
        options = self.settings.lake_storage_options or None
        return fsspec.core.url_to_fs(uri, **(options or {}))

    def save_pyfunc(
        self,
        *,
        name: str,
        run_id: str,
        model: Any,
        signature: Any = None,
        input_example: Any = None,
        pip_requirements: Sequence[str] | None = None,
    ) -> str:
        """Package ``model`` and upload it, publishing ``MLmodel`` last.

        The MLflow version tag is written by the registry only after this method
        succeeds, so readers never discover a partially uploaded model bundle.
        """
        uri = self.uri(name, run_id)
        fs, destination = self._filesystem(uri)
        with tempfile.TemporaryDirectory(prefix="maxxflow-model-") as temporary:
            local_model = Path(temporary) / "model"
            mlflow.pyfunc.save_model(
                path=str(local_model),
                python_model=model,
                signature=signature,
                input_example=input_example,
                pip_requirements=list(pip_requirements) if pip_requirements else None,
            )
            files = sorted(
                (path for path in local_model.rglob("*") if path.is_file()),
                key=lambda path: (path.name == "MLmodel", path.as_posix()),
            )
            for local_file in files:
                relative = local_file.relative_to(local_model).as_posix()
                remote_file = f"{destination.rstrip('/')}/{relative}"
                fs.makedirs(remote_file.rsplit("/", 1)[0], exist_ok=True)
                with local_file.open("rb") as source, fs.open(remote_file, "wb") as target:
                    shutil.copyfileobj(source, target)
        return uri

    def load_pyfunc(self, uri: str) -> Any:
        """Download one immutable bundle from the lake and load it locally."""
        fs, source = self._filesystem(uri)
        files = fs.find(source)
        marker = f"{source.rstrip('/')}/MLmodel"
        if marker not in files:
            raise FileNotFoundError(f"Data Lake model artifact is incomplete: {uri}")
        with tempfile.TemporaryDirectory(prefix="maxxflow-model-load-") as temporary:
            local_model = Path(temporary) / "model"
            for remote_file in files:
                if remote_file.endswith("/"):
                    continue
                relative = remote_file[len(source.rstrip('/')):].lstrip("/")
                local_file = local_model / relative
                local_file.parent.mkdir(parents=True, exist_ok=True)
                with fs.open(remote_file, "rb") as remote, local_file.open("wb") as local:
                    shutil.copyfileobj(remote, local)
            return mlflow.pyfunc.load_model(str(local_model))


@lru_cache(maxsize=32)
def load_lake_model(uri: str) -> Any:
    """Cache immutable versioned artifacts for the lifetime of the API process."""
    return ModelArtifactStore().load_pyfunc(uri)
