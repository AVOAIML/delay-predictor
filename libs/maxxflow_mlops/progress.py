"""Live training progress from a job the API cannot see.

THE PROBLEM
-----------
The thread backend shares a ``RunLogger`` list with the request handler, so the
Train page tails real lines as they happen. An Azure ML job runs in another
process on another machine, and the backend could only report the job's STATE —
"Queued", "Running", "Completed". A five-minute fit looked identical to a hung
one, and the only way to see what was happening was the Studio link.

WHY NOT READ AML'S LOGS FOR THIS
--------------------------------
``client.jobs.stream()`` blocks until the job ends, so it cannot back a polling
endpoint. ``client.jobs.download()`` works but pulls the job's whole artifact
tree from blob — far too heavy to run every two seconds, which is how often the
UI polls. Both also tie the progress feed to one backend.

WHAT THIS DOES INSTEAD
----------------------
The job writes its own lines to the SAME MLflow server that already holds the
model and ``configurator_result.json``. The Configurator reads them from there.
One store, one hop, and — the property that matters — the UI receives the
identical ``logs[]`` shape whichever backend ran the training. That is the same
reasoning that put the trainer's result in an MLflow artifact rather than in the
job's stdout.

Correlation is by a caller-supplied id, not by the AML job name: the backend
mints the id BEFORE submitting, so there is no window where a job is running and
its progress cannot be found. It is passed in as ``--progress-id``.

WHAT THIS DELIBERATELY DOES NOT COVER
-------------------------------------
Anything that fails before the job's Python starts — an image that will not
pull, a missing module, a bad managed identity — writes nothing here, because
nothing here is running yet. Those are exactly the failures worth seeing, so the
backend fetches the cluster's own ``user_logs/std_log.txt`` on terminal failure.
This file is the happy path; that is the post-mortem.
"""

from __future__ import annotations

import json
import time

from maxxflow_core.errors import get_logger

log = get_logger("mlops.progress")

EXPERIMENT = "m1-configurator-progress"
ARTIFACT = "progress.log"
STATE_ARTIFACT = "progress.json"
TAG = "maxxflow.progress_id"
_FLUSH_SECONDS = 3.0


def _client():
    from mlflow.tracking import MlflowClient

    from maxxflow_mlops.registry import MLflowRegistry

    reg = MLflowRegistry()
    return MlflowClient(reg.tracking_uri, reg.tracking_uri)


class ProgressPublisher:
    """Append-and-flush a line buffer to one MLflow run.

    ``log_text`` REWRITES the artifact, so every flush uploads the whole buffer.
    That is fine at this size (a training run produces tens of lines, not
    thousands) and it means a reader never sees a partial file — but it is why
    flushes are rate-limited rather than per-line.

    Nothing here may raise. This is a progress feed attached to a training run;
    losing it must never turn a model that trained into a run that failed. Every
    method swallows and logs, and the publisher marks itself dead after the first
    failure rather than retrying into a timeout on every line.
    """

    def __init__(self, progress_id: str, *, flush_seconds: float = _FLUSH_SECONDS) -> None:
        self.progress_id = progress_id
        self.lines: list[str] = []
        self.progress_state: dict | None = None
        self._flush_seconds = flush_seconds
        self._last_flush = 0.0
        self._run_id: str | None = None
        self._dead = False
        self._start()

    def _start(self) -> None:
        try:
            client = _client()
            exp = client.get_experiment_by_name(EXPERIMENT)
            exp_id = exp.experiment_id if exp else client.create_experiment(EXPERIMENT)
            run = client.create_run(exp_id, tags={TAG: self.progress_id,
                                                  "mlflow.runName": f"progress-{self.progress_id}"})
            self._run_id = run.info.run_id
        except Exception as e:
            self._dead = True
            log.warning("progress feed unavailable (%s: %s) — training continues, the UI "
                        "will show job state only", type(e).__name__, e)

    def append(self, line: str) -> None:
        if self._dead:
            return
        self.lines.append(line)
        if time.monotonic() - self._last_flush >= self._flush_seconds:
            self.flush()

    def update_progress(self, progress: dict) -> None:
        if self._dead:
            return
        self.progress_state = dict(progress)
        if time.monotonic() - self._last_flush >= self._flush_seconds:
            self.flush()

    def flush(self) -> None:
        if self._dead or self._run_id is None or (not self.lines and self.progress_state is None):
            return
        try:
            client = _client()
            if self.lines:
                client.log_text(self._run_id, "\n".join(self.lines), ARTIFACT)
            if self.progress_state is not None:
                client.log_text(self._run_id, json.dumps(self.progress_state), STATE_ARTIFACT)
            self._last_flush = time.monotonic()
        except Exception as e:
            self._dead = True
            log.warning("progress flush failed (%s: %s) — feed stops here, training "
                        "continues", type(e).__name__, e)

    def close(self, status: str = "FINISHED") -> None:
        """Final flush, then terminate the run.

        The final flush is unconditional: rate limiting means the last few lines
        — which include the promotion verdict, the most useful line in the whole
        feed — would otherwise never be uploaded.
        """
        self._last_flush = 0.0
        self.flush()
        if self._dead or self._run_id is None:
            return
        try:
            _client().set_terminated(self._run_id, status)
        except Exception as e:
            log.warning("could not terminate progress run (%s: %s)", type(e).__name__, e)


def read_progress(progress_id: str) -> list[str]:
    """The lines written so far, oldest first. Empty if the job has not started.

    Empty is the normal state for a queued job and is not an error: serverless
    compute spends its first minutes provisioning, during which nothing has run.
    """
    try:
        import os

        import mlflow

        # The Train page polls every two seconds, and load_text renders a tqdm
        # progress bar per call. Unsuppressed, that is thirty progress bars a
        # minute in the API's container log for a file of a few kilobytes.
        os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

        client = _client()
        exp = client.get_experiment_by_name(EXPERIMENT)
        if exp is None:
            return []
        runs = client.search_runs([exp.experiment_id],
                                  filter_string=f"tags.`{TAG}` = '{progress_id}'",
                                  max_results=1)
        if not runs:
            return []
        text = mlflow.artifacts.load_text(f"runs:/{runs[0].info.run_id}/{ARTIFACT}")
        return [ln for ln in text.splitlines() if ln]
    except Exception as e:
        # A missing artifact is the common case (run created, nothing flushed
        # yet) and must not surface as an error in the Train page.
        log.debug("no progress for %s (%s: %s)", progress_id, type(e).__name__, e)
        return []


def read_progress_state(progress_id: str) -> dict | None:
    """Read the latest structured progress snapshot for an Azure ML job."""
    try:
        import mlflow

        client = _client()
        exp = client.get_experiment_by_name(EXPERIMENT)
        if exp is None:
            return None
        runs = client.search_runs([exp.experiment_id],
                                  filter_string=f"tags.`{TAG}` = '{progress_id}'",
                                  max_results=1)
        if not runs:
            return None
        text = mlflow.artifacts.load_text(
            f"runs:/{runs[0].info.run_id}/{STATE_ARTIFACT}"
        )
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except Exception as e:
        log.debug("no progress state for %s (%s: %s)", progress_id, type(e).__name__, e)
        return None
