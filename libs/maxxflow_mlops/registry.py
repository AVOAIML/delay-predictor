"""MLflow registry adapter — the ``ModelRegistry`` port's local==Azure adapter.

HARD RULE (plan §2, §12a #1): route by registered-model NAME + ``@alias`` load
ONLY. This adapter deliberately exposes no search/stage methods. The parity test
``tests/parity/test_registry_name_only.py`` greps this package for
``search_model_versions`` / ``search_registered_models`` / ``get_latest_versions``
and fails the build if any appears — those work on OSS MLflow locally but
silently fail on Azure ML, so they must never enter the codebase.

Promotion = moving the ``@champion`` alias (no redeploy). Rollback =
``@champion`` -> ``@previous``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import mlflow
from mlflow.tracking import MlflowClient

from maxxflow_core.errors import RegistryRoutingError, get_logger
from maxxflow_core.settings import get_settings
from maxxflow_mlops.naming import GLOBAL_TENANT, global_model_name, registered_model_name

log = get_logger("maxxflow_mlops.registry")

# A small, explicit pip set so MLflow doesn't run its slow requirement inference.
_PYFUNC_PIP = [
    "mlflow",
    "scikit-learn",
    "imbalanced-learn",
    "lightgbm",
    "xgboost",
    "pandas",
    "numpy",
    "cloudpickle",
    "scipy",
]


def _resolve_uri() -> str:
    """Which MLflow to talk to, most-trusted source first.

    MAXXFLOW_MLFLOW_URI wins because MLFLOW_TRACKING_URI is not ours inside an
    Azure ML job: AML sets it to the workspace's own endpoint and that value
    overrode the one the job spec supplied, so training tried to register into a
    registry with no aliases. Anything the platform sets, the platform can
    change; a name it has never heard of, it cannot."""
    s = get_settings()
    if s.maxxflow_mlflow_uri:
        return s.maxxflow_mlflow_uri
    if s.mlflow_tracking_uri:
        return s.mlflow_tracking_uri
    # DB-less local default: sqlite registry (file store can't host the registry).
    repo_root = Path(__file__).resolve().parents[2]
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{repo_root / 'mlflow.db'}")


# Everything Azure ML injects into a job container to attach your process to ITS
# MLflow run. Harmless when you are using the workspace's tracking server; poison
# once you point somewhere else, because these ids and credentials refer to
# objects that exist only in AML's store.
_AMBIENT_MLFLOW_VARS = ("MLFLOW_RUN_ID", "MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME",
                        "MLFLOW_TRACKING_TOKEN", "MLFLOW_TRACKING_AUTH",
                        "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")


def detach_from_ambient_tracking() -> list[str]:
    """Drop the host platform's MLflow run context. Returns what it cleared.

    THE CLASS OF BUG THIS CLOSES. Azure ML does not merely set
    MLFLOW_TRACKING_URI in a job container — it seeds the whole run context, so
    that a bare ``mlflow.start_run()`` silently attaches to the AML run it created
    for the job. That is convenient if you use AML's tracking server and fatal if
    you do not: the ids name runs in a store we are not talking to, and the
    failure surfaces deep in the REST client as

        RESOURCE_DOES_NOT_EXIST: Run with id=<aml-run-name> not found

    which reads like our data is missing rather than like we inherited someone
    else's session. Fixing each variable where it bites means finding each one
    the hard way, in a five-minute-per-attempt cloud loop. Clearing the set here,
    once, in the only place that decides which registry we talk to, covers every
    entrypoint: CLI, API, scheduled job, and anything added later.

    Only runs when MAXXFLOW_MLFLOW_URI is set — i.e. when someone has deliberately
    said "use THIS registry, not the ambient one". Without that we leave the
    environment alone, so running under AML's own tracking still works normally.
    """
    if not os.environ.get("MAXXFLOW_MLFLOW_URI"):
        return []
    cleared = [v for v in _AMBIENT_MLFLOW_VARS if os.environ.pop(v, None) is not None]
    # A run opened before we switched stores belongs to the old one; ending it
    # here avoids logging half a run into each.
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception:
        pass
    return cleared


class MLflowRegistry:
    """Adapter implementing :class:`maxxflow_core.ports.ModelRegistry`."""

    # Registry backends MLflow can host ALIASES on. Azure ML's MLflow endpoint is
    # deliberately absent: it exposes tracking, and a model registry that supports
    # deprecated STAGES, not aliases. Everything below - set_alias,
    # get_model_version_by_alias, models:/name@champion, promote()'s
    # champion/previous swap - is alias-native, so pointing this at azureml://
    # cannot work. It fails today deep inside MlflowClient with
    # "UnsupportedModelRegistryStoreURIException", which reads like a missing
    # plugin rather than the wrong backend.
    _ALIAS_CAPABLE = ("http://", "https://", "sqlite:", "postgresql:", "mysql:",
                      "mssql:", "file:", "databricks")

    def __init__(self, tracking_uri: str | None = None):
        # Before anything reads the environment: shed the host platform's run
        # context, or mlflow.start_run() will try to resume a run that lives in a
        # different store. See detach_from_ambient_tracking.
        dropped = detach_from_ambient_tracking()
        if dropped:
            log.info("ignoring host-injected MLflow context: %s", ", ".join(dropped))
        self.tracking_uri = tracking_uri or _resolve_uri()
        if not self.tracking_uri.startswith(self._ALIAS_CAPABLE):
            raise RuntimeError(
                f"MLFLOW_TRACKING_URI is {self.tracking_uri!r}, which cannot host model "
                f"aliases. This codebase promotes by moving the @champion alias (see "
                f"maxxflow_mlops.promotion); Azure ML's registry offers deprecated stages "
                f"instead, so publish(), rollback and ModelRouter would all break.\n"
                f"Inside an Azure ML job this usually means AML's own tracking URI won "
                f"over the one you set: AML injects MLFLOW_TRACKING_URI into every job. "
                f"Pass the registry explicitly with `maxxflow train-csv --mlflow-uri "
                f"https://<your-mlflow-host>` so ambient environment cannot override it.")
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.tracking_uri)
        self.client = MlflowClient(self.tracking_uri, self.tracking_uri)

    # --- write path ----------------------------------------------------------
    def log_and_register(self, model: Any, *, name: str, params: Mapping[str, Any],
                         metrics: Mapping[str, float], tags: Mapping[str, str],
                         signature: Any = None, input_example: Any = None,
                         experiment: str | None = None) -> str:
        """Log an ``mlflow.pyfunc.PythonModel`` and register a new version.

        ``model`` must be a PythonModel instance defined in an importable module
        (so cloudpickle references it by path — uniform BYOC serving). Returns the
        new version number as a string.
        """
        mlflow.set_experiment(experiment or name)
        with mlflow.start_run() as run:
            if params:
                mlflow.log_params(dict(params))
            if metrics:
                mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            if tags:
                mlflow.set_tags(dict(tags))
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=model,
                signature=signature,
                input_example=input_example,
                pip_requirements=_PYFUNC_PIP,
            )
            model_uri = f"runs:/{run.info.run_id}/model"
            mv = mlflow.register_model(model_uri=model_uri, name=name, tags=dict(tags))
        return mv.version

    # --- alias ops (NO search, NO stages) ------------------------------------
    def set_alias(self, *, name: str, alias: str, version: str) -> None:
        self.client.set_registered_model_alias(name=name, alias=alias, version=str(version))

    def get_alias_version(self, *, name: str, alias: str) -> str | None:
        try:
            mv = self.client.get_model_version_by_alias(name=name, alias=alias)
            return mv.version
        except Exception:
            return None

    def get_alias_tags(self, *, name: str, alias: str) -> dict:
        mv = self.client.get_model_version_by_alias(name=name, alias=alias)
        return dict(mv.tags or {})

    # --- read path (load by name@alias ONLY) ---------------------------------
    def load_champion(self, *, name: str, alias: str | None = None) -> Any:
        alias = alias or get_settings().model_alias
        uri = f"models:/{name}@{alias}"
        return mlflow.pyfunc.load_model(uri)

    def resolve_champion_name(self, *, tenant: str, module: str,
                              alias: str | None = None) -> tuple[str, bool] | None:
        """Return ``(name, is_base)`` of the champion that should serve this
        (tenant, module): the tenant's OWN model if it has a champion, otherwise the
        shared GLOBAL base model. ``None`` if neither exists yet.

        This is the onboarding fallback — a brand-new tenant with no trained model is
        served the base model (``t_global__m_<module>``) until it publishes its own.
        Uses ``@alias`` existence checks only (no search — plan §12a #1).
        """
        alias = alias or get_settings().model_alias
        own = registered_model_name(tenant, module)
        if self.get_alias_version(name=own, alias=alias) is not None:
            return own, False
        if tenant != GLOBAL_TENANT:
            base = global_model_name(module)
            if self.get_alias_version(name=base, alias=alias) is not None:
                return base, True
        return None

    def promote(self, *, name: str, challenger_version: str, alias: str | None = None) -> None:
        """Champion/challenger promotion = alias move; keeps a ``previous`` alias
        for sub-minute rollback (plan §6). Gate logic lives in ``promotion.py``."""
        alias = alias or get_settings().model_alias
        current = self.get_alias_version(name=name, alias=alias)
        if current is not None:
            self.client.set_registered_model_alias(name=name, alias="previous", version=current)
        self.client.set_registered_model_alias(name=name, alias=alias, version=str(challenger_version))

    # Guard: explicitly forbid the unsupported search pattern if ever called.
    def search(self, *args, **kwargs):  # pragma: no cover - intentional tripwire
        raise RegistryRoutingError(
            "tag/alias search is unsupported on Azure ML's MLflow registry; "
            "route by registered-model NAME + @alias load only (plan §12a #1)."
        )


def get_registry() -> MLflowRegistry:
    return MLflowRegistry()
