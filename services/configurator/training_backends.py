"""Where a training run actually executes.

Two backends behind one interface, selected by ``TRAIN_BACKEND`` in Settings —
the same ports-and-factory shape the rest of the repo uses, so no module ever
compares the environment (the parity guard greps for that).

    thread    the run happens in a Python thread inside this API process
    azureml   the run happens on Azure ML serverless compute

WHY BOTH, RATHER THAN JUST MOVING TO AML
----------------------------------------
The thread backend is not legacy. It is what makes `make ui` work on a laptop
with no Azure at all, and what lets the test suite train without a workspace.
Deleting it would mean every contributor needs an AML subscription to run the
UI, which is a worse trade than keeping a second small implementation.

WHAT THE AML BACKEND FIXES
--------------------------
The thread backend trains inside ca-configurator, on the 1 CPU / 2 GiB that is
also serving the page you are watching. A long fit starves request handling, a
revision restart kills the run with no record, and it is the reason the app must
be pinned to a single replica: uploads land on container-local disk and run
state lives in a module-level dict, so a second replica sees neither.

The AML backend has none of those properties — the fit gets its own VM, the CSV
goes to blob, and the run's state lives in AML. It does NOT by itself make the
API replica-safe (the run_id -> job-name mapping below is still in memory), but
it removes the compute contention and the mid-training data loss, which are the
parts that actually bite.

RESULT RETRIEVAL
----------------
An AML run's outcome is read back from MLflow, not from the job's logs. The job
registers the candidate in the SAME self-hosted registry the Publish button
reads, so asking MLflow "what is the newest version of this model" is both the
robust answer (no log parsing, no dependence on log retention) and the honest
one — if the version is not in the registry, there is nothing to publish and the
run should not claim success.
"""

from __future__ import annotations

import time
import uuid
from typing import Protocol

from maxxflow_core.errors import get_logger
from maxxflow_core.settings import get_settings
from maxxflow_mlops.naming import registered_model_name
from services.configurator.result_store import get_training_result_store

log = get_logger("configurator.training_backends")

# AML terminal/active states -> the three the UI already understands. The UI polls
# for "running" and renders "done"/"error"; teaching it a fourth vocabulary would
# mean changing the frontend for an implementation detail of the backend.
_AML_STATUS = {
    "NotStarted": "running", "Starting": "running", "Provisioning": "running",
    "Preparing": "running", "Queued": "running", "Running": "running",
    "Finalizing": "running", "CancelRequested": "running",
    "Completed": "done", "Failed": "error", "Canceled": "error", "NotResponding": "error",
}


class TrainingBackend(Protocol):
    def start(self, tenant: str, model_key: str, *, source: str, csv_path: str | None,
              auto_hpo: bool, data_tenant: str | None = None,
              source_name: str | None = None,
              dataset_row_count: int | None = None) -> str: ...
    def status(self, run_id: str) -> dict: ...
    def active_runs(self, tenant: str) -> list[dict]: ...
    def pending_runs(self, tenant: str) -> list[dict]: ...
    def mark_published(self, tenant: str, model_key: str, version: str) -> None: ...


class AzureMLBackend:
    """Submit `maxxflow train-csv` as an Azure ML command job.

    Deliberately does NOT pass --publish. The Configurator's whole point is that
    a human reviews the candidate's metrics and then decides; auto-publishing
    from the UI path would take that decision away and make the Publish button a
    lie. The SCHEDULED job (infra/aml/retrain-job.yml) does pass --publish,
    because there is nobody to review it — different context, different default.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._runs: dict[str, dict] = {}

    def _client(self):
        s = self.settings
        # Config FIRST. Checking after the import means an operator who set
        # TRAIN_BACKEND=azureml and nothing else gets ModuleNotFoundError, which
        # points at a missing package rather than at the four settings they
        # actually forgot.
        missing = [n for n, v in (("AML_SUBSCRIPTION_ID", s.aml_subscription_id),
                                  ("AML_RESOURCE_GROUP", s.aml_resource_group),
                                  ("AML_WORKSPACE", s.aml_workspace),
                                  ("AML_TRAIN_IMAGE", s.aml_train_image)) if not v]
        if missing:
            raise RuntimeError(
                f"TRAIN_BACKEND=azureml but {', '.join(missing)} "
                f"{'is' if len(missing) == 1 else 'are'} not set — nothing to submit to. "
                f"Set TRAIN_BACKEND=thread to train in-process instead.")
        # Imported lazily: azure-ai-ml is the optional `aml` extra, and the
        # thread backend must keep working in an image without it.
        try:
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential
        except ImportError as e:
            raise RuntimeError(
                "TRAIN_BACKEND=azureml needs the `aml` extra (azure-ai-ml, "
                "azure-identity), which is not installed in this image. Rebuild with "
                "`uv sync --extra aml`, or set TRAIN_BACKEND=thread.") from e
        return MLClient(DefaultAzureCredential(), s.aml_subscription_id,
                        s.aml_resource_group, s.aml_workspace)

    def start(self, tenant: str, model_key: str, *, source: str, csv_path: str | None,
              auto_hpo: bool, data_tenant: str | None = None,
              source_name: str | None = None,
              dataset_row_count: int | None = None) -> str:
        if source == "csv" and not csv_path:
            raise ValueError("source='csv' needs a csv_path")
        if source not in ("csv", "db"):
            raise ValueError(f"unknown source {source!r}")
        # _client() first: it owns BOTH the config check and the "is the SDK even
        # installed" check, and both produce an actionable message. Importing
        # azure.ai.ml above this line put a bare ModuleNotFoundError in front of
        # them, so a missing setting reported itself as a missing package.
        client = self._client()
        from azure.ai.ml import Input, command
        from azure.ai.ml.entities import Environment, ManagedIdentityConfiguration
        s = self.settings
        s_uri = s.mlflow_tracking_uri

        # The local upload is handed to AML as a job input. The SDK uploads it to
        # the workspace datastore, which is what makes this work at all: AML
        # compute cannot see this container's filesystem.
        # --mlflow-uri, not just the env var. AML injects its own
        # MLFLOW_TRACKING_URI into every job and it WON over ours on the first
        # real run: the job tried to use the workspace's registry, which has no
        # aliases, and died inside MlflowClient. A CLI argument cannot be
        # overridden by the platform.
        # Minted HERE, before submitting, and passed as --progress-id. Correlating
        # on the AML job name would leave a window between "the job exists" and
        # "this process knows its name" in which a running job's progress could
        # not be looked up. See libs/maxxflow_mlops/progress.py.
        progress_id = uuid.uuid4().hex[:12]
        resolved_source_name = source_name or (
            "MaXXflow Database" if source == "db" else str(csv_path).rsplit("/", 1)[-1]
        )
        if source == "csv":
            cmd = ["maxxflow train-csv", f"--model {model_key}", "--csv ${{inputs.csv}}",
                   f"--tenant {tenant}", f"--mlflow-uri {s_uri}",
                   f"--progress-id {progress_id}"]
            inputs = {"csv": Input(type="uri_file", path=csv_path, mode="ro_mount")}
        else:
            # No job input: the rows are in Postgres, and the job reaches it
            # directly. --data-tenant is the SCHEMA; --tenant names the model.
            cmd = ["maxxflow train-db", f"--model {model_key}",
                   f"--tenant {tenant}",
                   f"--data-tenant {data_tenant or tenant}",
                   f"--mlflow-uri {s_uri}",
                   f"--progress-id {progress_id}"]
            inputs = {}
        if not auto_hpo:
            cmd.append("--no-hpo")

        job = command(
            display_name=f"configurator-{model_key}-{source}",
            experiment_name="m1-configurator-retrain",
            command=" ".join(cmd),
            inputs=inputs,
            # An Environment OBJECT, not the image string. A bare string here is
            # parsed as a reference to a REGISTERED environment ("azureml:<name>"),
            # so passing a container image produced:
            #   Value passed is not a data binding string:
            #   azureml:cravonetm1.azurecr.io/maxxflow:train-latest
            # Wrapping it declares an anonymous environment built on that image,
            # which is what BYOC means — our own container, our own pinned env.
            environment=Environment(image=s.aml_train_image),
            environment_variables={
                "APP_ENV": s.app_env,
                # The candidate must land in OUR registry, not the workspace's.
                # AML injects its own MLFLOW_TRACKING_URI; this overrides it.
                # Azure ML's registry supports deprecated stages, not the aliases
                # publish() moves, so a run that logged there would register a
                # model the Publish button cannot promote.
                # AML overwrites MLFLOW_TRACKING_URI in the container; it does not
                # touch MAXXFLOW_-prefixed names, and _resolve_uri prefers this.
                "MAXXFLOW_MLFLOW_URI": s.mlflow_tracking_uri,
                "MLFLOW_TRACKING_URI": s.mlflow_tracking_uri,
                "MODEL_ALIAS": s.model_alias,
                # Model bytes are written directly to the tenant Data Lake before
                # the MLflow version metadata is created. For ADLS, the declared
                # job identity must have Storage Blob Data Contributor access.
                "LAKE_URI": s.lake_uri,
                "MAXXFLOW_DATA_SOURCE_NAME": resolved_source_name,
                **({"MAXXFLOW_DATASET_ROW_COUNT": str(dataset_row_count)}
                   if dataset_row_count is not None else {}),
                # Postgres connection PARTS, and the LOCATION of the password.
                # PGPASS is deliberately absent: environment_variables are stored
                # with the job and shown in Studio, so a password here would
                # persist in the history of every run and outlive its rotation.
                # The job fetches it from Key Vault with the compute's managed
                # identity (libs/maxxflow_core/keyvault.py). Empty values are
                # dropped below so a CSV job in a DB-less deployment is unchanged.
                **{k: v for k, v in {
                    "PGHOST": s.pg_host,
                    "PGPORT": str(s.pg_port),
                    "PGUSER": s.pg_user,
                    "PGDATABASE": s.pg_database,
                    "PGSSLMODE": s.pg_sslmode,
                    "KEYVAULT_NAME": s.keyvault_name,
                    "PG_PASSWORD_SECRET": s.pg_password_secret,
                    # Which identity DefaultAzureCredential should use. Without
                    # it the chain tries the system-assigned one first and gets a
                    # 400 from IMDS on a box that only has a user-assigned.
                    "AZURE_CLIENT_ID": s.aml_job_identity_client_id,
                }.items() if v},
            },
            instance_type=s.aml_instance_type,
            # Serverless compute runs with an AML TOKEN unless the job declares
            # otherwise. That token is scoped to the workspace, so a job that has
            # to read Key Vault (the DB path fetches PGPASS there) finds no
            # managed identity at all and DefaultAzureCredential dies on IMDS
            # with "Expecting value: line 1 column 1" — a JSON parse error that
            # looks like a library bug. Declaring the user-assigned identity is
            # what actually puts one on the box.
            #
            # Left unset when AML_JOB_IDENTITY_CLIENT_ID is empty, which keeps the
            # CSV path exactly as it was: it needs nothing outside the workspace.
            **({"identity": ManagedIdentityConfiguration(
                    client_id=s.aml_job_identity_client_id)}
               if s.aml_job_identity_client_id else {}),
        )
        # Record the registry high-water mark BEFORE submitting. Without it,
        # "newest version" is ambiguous: a job that completes without registering
        # anything would hand back the PREVIOUS candidate, and the UI would show
        # someone else's metrics next to a Publish button as if this run produced
        # them. The result is only accepted if the version actually moved.
        version_before = _latest_version_number(tenant, model_key)
        submitted = client.jobs.create_or_update(job)
        run_id = submitted.name
        self._runs[run_id] = {"tenant": tenant, "model_key": model_key, "source": source,
                              "data_tenant": data_tenant or tenant,
                              "source_name": resolved_source_name,
                              "dataset_row_count": dataset_row_count,
                              "started_at": time.time(), "studio_url": _studio_url(submitted),
                              "version_before": version_before,
                              "progress_id": progress_id, "failure_log": None,
                              "last_status": "running", "published": False,
                              "result_version": None}
        log.info("submitted AML job %s for %s/%s", run_id, tenant, model_key)
        return run_id

    def status(self, run_id: str) -> dict:
        meta = self._runs.get(run_id)
        if meta is None:
            # The API restarted, or another replica submitted it. AML still knows
            # about the job, so report what it says rather than "unknown" — the
            # run is real even when this process has forgotten submitting it.
            meta = {"tenant": "?", "model_key": "?", "source": "csv",
                    "data_tenant": "?",
                    "started_at": time.time(), "studio_url": None}
        try:
            job = self._client().jobs.get(run_id)
        except Exception as e:
            meta["last_status"] = "error"
            meta.setdefault("finished_at", time.time())
            return {"run_id": run_id, "status": "error",
                    "error": f"could not reach Azure ML: {type(e).__name__}: {e}",
                    "logs": [],
                    "progress": {"percent": 0, "phase": "error",
                                 "label": "Could not reach Azure ML",
                                 "current": None, "total": None},
                    **_common(meta, run_id)}

        status = _AML_STATUS.get(job.status, "running")
        if status != "running":
            meta.setdefault("finished_at", time.time())
        # Header first, then the job's OWN lines. The header is the platform's
        # view (queued / provisioning / running); the feed is the training's view.
        # A queued job legitimately has no feed yet — serverless spends its first
        # minutes getting a VM — so an empty feed is not an error and says so.
        header = [f"Azure ML job {run_id}: {job.status}"]
        if meta.get("studio_url"):
            header.append(f"Studio: {meta['studio_url']}")
        feed = _progress_lines(meta.get("progress_id"))
        progress = _progress_state(meta.get("progress_id"))
        if progress is None:
            if status == "done":
                progress = {"percent": 100, "phase": "complete", "label": "Training complete",
                            "current": None, "total": None}
            elif status == "error":
                progress = {"percent": 0, "phase": "error", "label": f"Azure ML: {job.status}",
                            "current": None, "total": None}
            else:
                progress = {
                    "percent": 0,
                    "phase": "queued" if job.status != "Running" else "starting",
                    "label": f"Azure ML: {job.status}",
                    "current": None,
                    "total": None,
                }
        if not feed and status == "running":
            header.append("waiting for the job to start — serverless compute provisions "
                          "a VM first, so there is nothing to report yet")
        out = {"status": status, "logs": header + feed, "progress": progress,
               **_common(meta, run_id)}
        if status == "done":
            result = _latest_registered(meta["tenant"], meta["model_key"],
                                        after=meta.get("version_before", 0))
            if result is None:
                # Completed, but nothing landed in the registry. Saying "done"
                # here would put a Publish button in front of a model that does
                # not exist.
                out["status"] = "error"
                out["error"] = ("the job completed but no new model version appeared in "
                                f"MLflow at {self.settings.mlflow_tracking_uri} — check that "
                                "MLFLOW_TRACKING_URI reached the job and was not overridden "
                                "by the workspace's own tracking URI")
            else:
                out["result"] = result
                meta["result_version"] = str(result["version"])
                out["published"] = bool(meta.get("published", False))
                get_training_result_store().save(out)
        elif status == "error":
            out["error"] = f"Azure ML job {run_id} finished as {job.status}"
            # The feed cannot explain a failure that happened BEFORE the job's
            # Python ran — an image that will not pull, a missing module, an
            # identity that cannot fetch the image. Those live only in the
            # cluster's own stdout. Fetch it once, here, on the terminal poll:
            # jobs.download() pulls the job's artifact tree from blob and is far
            # too heavy to run on every two-second poll, which is why it is
            # cached on meta rather than re-fetched.
            if meta.get("failure_log") is None:
                meta["failure_log"] = self._cluster_log_tail(run_id)
            if meta["failure_log"]:
                out["logs"] = out["logs"] + ["", "--- cluster log (last lines) ---",
                                             *meta["failure_log"]]
        meta["last_status"] = out["status"]
        return out

    def active_runs(self, tenant: str) -> list[dict]:
        """Refresh and return this process's currently active tenant runs."""
        candidates = [
            (run_id, meta) for run_id, meta in self._runs.items()
            if meta["tenant"] == tenant and meta.get("last_status") == "running"
        ]
        candidates.sort(key=lambda item: item[1]["started_at"], reverse=True)
        active = []
        for run_id, _ in candidates:
            current = self.status(run_id)
            if current.get("status") == "running":
                active.append(current)
        return active

    def pending_runs(self, tenant: str) -> list[dict]:
        """Refresh running jobs and retain completed candidates until publication."""
        candidates = [
            (run_id, meta) for run_id, meta in self._runs.items()
            if meta["tenant"] == tenant
            and meta.get("last_status") in ("running", "done")
            and not meta.get("published", False)
        ]
        candidates.sort(key=lambda item: item[1]["started_at"], reverse=True)
        pending = []
        for run_id, _ in candidates:
            current = self.status(run_id)
            if current.get("status") in ("running", "done"):
                pending.append(current)
        return pending

    def mark_published(self, tenant: str, model_key: str, version: str) -> None:
        matching_started_at = next((
            meta["started_at"] for meta in self._runs.values()
            if meta["tenant"] == tenant and meta["model_key"] == model_key
            and str(meta.get("result_version")) == str(version)
        ), None)
        if matching_started_at is None:
            return
        for meta in self._runs.values():
            if (meta["tenant"] == tenant and meta["model_key"] == model_key
                    and meta["started_at"] <= matching_started_at):
                meta["published"] = True

    def _cluster_log_tail(self, run_id: str, *, lines: int = 40) -> list[str]:
        """The tail of the job's std_log.txt, or a note saying why there is none.

        There is no "give me the log text" call in the v2 SDK. jobs.stream()
        blocks until the job finishes, so it cannot serve a polling endpoint;
        download() is the sanctioned alternative. It writes the whole artifact
        tree, so this runs in a temp directory that is discarded immediately.

        Returns [] rather than raising: a missing log is a worse UI than a bare
        error message, but a 500 while rendering one is worse still.
        """
        import glob
        import tempfile

        try:
            with tempfile.TemporaryDirectory() as tmp:
                self._client().jobs.download(name=run_id, download_path=tmp, all=True)
                # The path has moved between SDK versions (user_logs/,
                # azureml-logs/, artifacts/user_logs/), so search rather than
                # assume. std_log.txt is the driver's combined stdout/stderr.
                hits = sorted(glob.glob(f"{tmp}/**/std_log*.txt", recursive=True)) or \
                    sorted(glob.glob(f"{tmp}/**/*.txt", recursive=True))
                if not hits:
                    return ["(no std_log.txt in the job's artifacts — the failure is "
                            "probably before the container started; open Studio)"]
                with open(hits[0], errors="replace") as fh:
                    tail = fh.read().splitlines()[-lines:]
                return [ln for ln in tail if ln.strip()]
        except Exception as e:
            log.warning("could not download logs for %s (%s: %s)", run_id, type(e).__name__, e)
            return [f"(could not fetch the cluster log: {type(e).__name__}: {e} — "
                    f"the Studio link above has it)"]


def _progress_lines(progress_id: str | None) -> list[str]:
    """The job's own log lines, read from MLflow. Never raises: a progress feed
    that fails must not take the status endpoint down with it."""
    if not progress_id:
        return []
    try:
        from maxxflow_mlops.progress import read_progress
        return read_progress(progress_id)
    except Exception as e:
        log.debug("progress unavailable for %s (%s: %s)", progress_id, type(e).__name__, e)
        return []


def _progress_state(progress_id: str | None) -> dict | None:
    if not progress_id:
        return None
    try:
        from maxxflow_mlops.progress import read_progress_state
        return read_progress_state(progress_id)
    except Exception as e:
        log.debug("progress state unavailable for %s (%s: %s)",
                  progress_id, type(e).__name__, e)
        return None


def _common(meta: dict, run_id: str) -> dict:
    elapsed_until = meta.get("finished_at") or time.time()
    source = meta["source"]
    kind = "database" if source == "db" else "csv"
    return {"run_id": run_id, "tenant": meta["tenant"], "model_key": meta["model_key"],
            "source": source,
            "data_source": {
                "kind": kind,
                "name": meta.get("source_name") or (
                    "MaXXflow Database" if kind == "database" else "CSV Dataset"
                ),
                "row_count": meta.get("dataset_row_count"),
            },
            "elapsed_s": round(elapsed_until - meta["started_at"], 1),
            "started_at": meta["started_at"],
            "finished_at": meta.get("finished_at")}


def _studio_url(job) -> str | None:
    try:
        return job.services["Studio"].endpoint
    except Exception:
        return None


def _versions(tenant: str, model_key: str):
    from mlflow.tracking import MlflowClient

    from maxxflow_mlops.registry import MLflowRegistry

    reg = MLflowRegistry()
    client = MlflowClient(reg.tracking_uri, reg.tracking_uri)
    name = registered_model_name(tenant, model_key)
    return client, client.search_model_versions(f"name=\'{name}\'")


def _latest_version_number(tenant: str, model_key: str) -> int:
    """Highest registered version right now, or 0 if the model is new."""
    try:
        _, versions = _versions(tenant, model_key)
    except Exception:
        return 0
    return max((int(v.version) for v in versions), default=0)


def _latest_registered(tenant: str, model_key: str, *, after: int = 0) -> dict | None:
    """The version THIS run produced, with the metrics from its run.

    ``after`` is the high-water mark taken at submission. Returning the newest
    version unconditionally would be wrong in the one case that matters: a job
    that finishes without registering anything (MLFLOW_TRACKING_URI overridden,
    training raised past the fit, gate crashed) would hand back the previous
    candidate, and the UI would offer someone else's metrics for publishing.

    The Publish button needs {version, metrics}; both come from the registry the
    job wrote to, which is the same one Publish reads. That is why this reads the
    registry rather than parsing the job's stdout: one source of truth, and no
    dependence on log formatting or retention."""
    try:
        client, versions = _versions(tenant, model_key)
    except Exception:
        return None
    if not versions:
        return None
    newest = max(versions, key=lambda v: int(v.version))
    if int(newest.version) <= after:
        return None

    # The trainer attaches its own result to the run (see the CLI's
    # _publish_result_artifact). Prefer it: the UI expects the SHAPE the
    # in-process backend returns — features, confusion, served_mode — and got a
    # white screen on `result.features.length` when handed metrics alone. Two
    # backends must not hand the frontend two different objects.
    try:
        import json

        import mlflow
        text = mlflow.artifacts.load_text(
            f"runs:/{newest.run_id}/configurator_result.json")
        result = json.loads(text)
        result["version"] = newest.version      # authoritative, whatever the file says
        return result
    except Exception as e:
        log.warning("no configurator_result.json on run %s (%s: %s) — falling back to "
                    "metrics only; the results panel will be sparse",
                    newest.run_id, type(e).__name__, e)
    run = client.get_run(newest.run_id)
    return {"version": newest.version, "metrics": dict(run.data.metrics),
            # Shape-compatible defaults so a missing artifact degrades to a thin
            # panel rather than an exception in the browser.
            "features": [], "source": "csv", "confusion": None, "champion": None}


def get_training_backend() -> TrainingBackend:
    """Settings-selected, never env-compared. A dict, so adding a backend is a
    row here and not a branch somewhere in a request handler."""
    from services.configurator import jobs

    name = get_settings().train_backend
    backends = {"thread": _ThreadBackend, "azureml": AzureMLBackend}
    if name not in backends:
        raise ValueError(f"unknown TRAIN_BACKEND {name!r}; choose from {sorted(backends)}")
    if name == "thread":
        return _ThreadBackend(jobs)
    return _singleton_azureml()


class _ThreadBackend:
    """The original in-process behaviour, unchanged, behind the same interface."""

    def __init__(self, jobs_module) -> None:
        self._jobs = jobs_module

    def start(self, tenant: str, model_key: str, *, source: str, csv_path: str | None,
              auto_hpo: bool, data_tenant: str | None = None,
              source_name: str | None = None,
              dataset_row_count: int | None = None) -> str:
        return self._jobs.start(tenant, model_key, source=source, csv_path=csv_path,
                                auto_hpo=auto_hpo, data_tenant=data_tenant,
                                source_name=source_name,
                                dataset_row_count=dataset_row_count)

    def status(self, run_id: str) -> dict:
        return self._jobs.status(run_id)

    def active_runs(self, tenant: str) -> list[dict]:
        return self._jobs.active_runs(tenant)

    def pending_runs(self, tenant: str) -> list[dict]:
        return self._jobs.pending_runs(tenant)

    def mark_published(self, tenant: str, model_key: str, version: str) -> None:
        self._jobs.mark_published(tenant, model_key, version)


_AML_SINGLETON: AzureMLBackend | None = None


def _singleton_azureml() -> AzureMLBackend:
    """One instance, because it holds the run_id -> job mapping. A fresh backend
    per request would forget every submission it had just made."""
    global _AML_SINGLETON
    if _AML_SINGLETON is None:
        _AML_SINGLETON = AzureMLBackend()
    return _AML_SINGLETON
