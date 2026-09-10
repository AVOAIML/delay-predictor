"""In-process training-job registry for the Configurator "Train" step.

Runs a training in a background thread (CSV upload OR read-replica DB source) and
exposes its LIVE log lines + result so the UI can poll and render the training log
(the RunLogger list is shared, so lines appear as they happen).

Both M1 models' CSV upload is the RAW combined export (one row per quotation line —
same shape as dataset/row/quotations_combined.csv), not a derived gold_*.csv shape:
it's run through m1_quote.raw_ingest here, server-side, before training — the client
only ever uploads/sees raw columns (services/configurator/app.py's preview validates
against raw_ingest.RAW_REQUIRED_COLS and REPORTS — never enforces — RAW_OPTIONAL_COLS,
so an export with no revision workflow or no negotiated-price column still trains;
raw_ingest documents what each absent optional column falls back to. Works with both
dataset/row/quotations_combined.csv and the wider
dataset/row/synthetic_quotations_all_verticals.csv."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

import pandas as pd

from m1_quote import csv_line_win, csv_mil, csv_price, csv_win, db_features
from m1_quote.csv_common import RunLogger
# The frame builders live in the library, not here, so the CLI and any Azure ML
# job run the identical conversion without importing the web service. Re-exported
# for callers that still reach for jobs.build_training_frame.
from m1_quote.frames import build_training_frame, write_option_graph  # noqa: F401
from m2_inventory.csv_training import train_for_configurator as train_inventory_csv
from m2_inventory.db_training import build_training_frame as build_inventory_db_frame
from maxxflow_mlops.registry import MLflowRegistry

_INVENTORY_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "m2_inventory"

_JOBS: dict[str, dict] = {}
_TRAINERS = {"m1_quote_win": csv_win.train, "m1_quote_price": csv_price.train,
             "m1_quote_line_win": csv_line_win.train, "m1_quote_mil": csv_mil.train,
             "m2_inventory": train_inventory_csv}
# m1_quote_mil is CSV-upload only (no DB-path builder yet) — absent from this
# dict, so a source="db" request for it fails naturally with a KeyError caught
# below, same as any other model without DB parity.
_DB_BUILDERS = {"m1_quote_win": db_features.build_win_frame,
                "m1_quote_price": db_features.build_price_frame,
                "m1_quote_line_win": db_features.build_line_frame,
                "m2_inventory": build_inventory_db_frame}


def db_models() -> list[str]:
    """Model keys with a DB-path builder. m1_quote_mil is CSV-upload only."""
    return sorted(_DB_BUILDERS)


def build_db_frame(tenant: str, model_key: str):
    """The frame source='db' training will consume — the SAME call, not a copy.

    The Configurator's DB preview goes through here rather than re-issuing the
    DAL SELECTs itself, because a preview assembled separately from the trainer
    is a preview that can disagree with it. Anything the user sees in that table
    is, by construction, the rows the model is about to fit.
    """
    if model_key not in _DB_BUILDERS:
        raise KeyError(model_key)
    return _DB_BUILDERS[model_key](tenant)


def start(tenant: str, model_key: str, *, source: str = "csv", csv_path: str | None = None,
          auto_hpo: bool = True, data_tenant: str | None = None,
          source_name: str | None = None,
          dataset_row_count: int | None = None) -> str:
    """`tenant` owns the model (it becomes the registered name). `data_tenant` is
    whose schema the DB path reads. They are the same for a tenant-owned model and
    differ for the shared base model, which is owned by `global` and has no schema
    of its own. Defaults to tenant, so the CSV path is unaffected."""
    if model_key not in _TRAINERS:
        raise KeyError(model_key)
    run_id = uuid.uuid4().hex[:12]
    read_from = data_tenant or tenant
    logger = RunLogger()
    logger.set_progress(0, "queued", "Waiting to start")
    resolved_source_name = source_name or (
        "MaXXflow Database" if source == "db" else str(csv_path).rsplit("/", 1)[-1]
    )
    job = {"run_id": run_id, "tenant": tenant, "model_key": model_key, "source": source,
           "data_tenant": read_from,
           "source_name": resolved_source_name, "dataset_row_count": dataset_row_count,
           "fallback_used": False, "fallback_reason": None,
           "status": "running", "logger": logger, "result": None, "error": None,
           "started_at": time.time(), "finished_at": None, "published": False}
    _JOBS[run_id] = job

    def _run():
        try:
            logger.set_progress(5, "preparing_data", "Preparing training data",
                                message="Training worker started")
            if model_key == "m2_inventory":
                dataset = csv_path
                if source == "db":
                    logger.log(f"Reading inventory snapshot history from tenant_{read_from}…")
                    dataset = _DB_BUILDERS[model_key](read_from)
                    job["dataset_row_count"] = int(len(dataset))
                    job["fallback_used"] = bool(dataset.attrs.get("fallback_used"))
                    job["fallback_reason"] = dataset.attrs.get("fallback_reason")
                    job["source_name"] = dataset.attrs.get("source_name", job["source_name"])
                    if job["fallback_used"]:
                        logger.log(
                            "WARNING: Real snapshot history is unavailable. Training with "
                            "generated bootstrap history; metrics are not evidence of real "
                            "historical performance."
                        )
                job["result"] = train_inventory_csv(
                    dataset, tenant, artifact_dir=_INVENTORY_ARTIFACTS,
                    register=True, source=source, logger=logger
                )
            elif source == "db":
                trainer = _TRAINERS[model_key]
                kwargs = {"logger": logger, "register": True, "source": source}
                if model_key in ("m1_quote_win", "m1_quote_price", "m1_quote_line_win"):
                    kwargs["auto_hpo"] = auto_hpo
                where = (f"schema tenant_{read_from} (this is the shared base model, "
                         f"owned by '{tenant}')") if read_from != tenant else f"schema tenant_{read_from}"
                logger.log(f"Connecting to the read replica and building features from {where}…")
                frame = _DB_BUILDERS[model_key](read_from)
                job["dataset_row_count"] = int(len(frame))
                job["result"] = trainer(frame, tenant, **kwargs)
            else:
                trainer = _TRAINERS[model_key]
                kwargs = {"logger": logger, "register": True, "source": source}
                if model_key in ("m1_quote_win", "m1_quote_price", "m1_quote_line_win"):
                    kwargs["auto_hpo"] = auto_hpo
                raw = pd.read_csv(csv_path)
                write_option_graph(raw, logger)
                frame = build_training_frame(model_key, raw, tenant, logger)
                job["result"] = trainer(frame, tenant, **kwargs)
            result = job["result"]
            result["fallback_used"] = job["fallback_used"]
            result["fallback_reason"] = job["fallback_reason"]
            MLflowRegistry().set_version_tags(
                name=result["registered_name"],
                version=str(result["version"]),
                tags={
                    "data_source_kind": "database" if source == "db" else "csv",
                    "data_source_name": job["source_name"],
                    "dataset_row_count": job["dataset_row_count"],
                    "algorithm": result.get("selected_model") or (
                        "lightgbm-isotonic" if model_key == "m1_quote_line_win" else None
                    ),
                    "fallback_used": str(job["fallback_used"]).lower(),
                    "fallback_reason": job["fallback_reason"],
                },
            )
            job["status"] = "done"
            job["finished_at"] = time.time()
            logger.log("Training complete — candidate registered. Review results, then Publish.")
            logger.set_progress(100, "complete", "Training complete")
        except Exception as e:
            job["error"] = f"{type(e).__name__}: {e}"
            job["status"] = "error"
            job["finished_at"] = time.time()
            logger.log(f"ERROR: {job['error']}")
            current = logger.progress_snapshot()
            logger.set_progress(current["percent"], "error", "Training failed")

    threading.Thread(target=_run, daemon=True).start()
    return run_id


def status(run_id: str) -> dict:
    job = _JOBS.get(run_id)
    if job is None:
        return {"status": "unknown"}
    elapsed_until = job.get("finished_at") or time.time()
    out = {"run_id": run_id, "tenant": job["tenant"], "model_key": job["model_key"],
           "source": job["source"], "status": job["status"], "logs": list(job["logger"].lines),
           "data_source": {
               "kind": "database" if job["source"] == "db" else "csv",
               "name": job["source_name"],
               "row_count": job["dataset_row_count"],
               "fallback_used": job["fallback_used"],
               "fallback_reason": job["fallback_reason"],
           },
           "elapsed_s": round(elapsed_until - job["started_at"], 1),
           "started_at": job["started_at"],
           "progress": job["logger"].progress_snapshot()}
    if job["status"] == "done":
        r = dict(job["result"]); r.pop("_model", None); r.pop("logs", None)
        out["result"] = r
    if job["status"] == "error":
        out["error"] = job["error"]
    return out


def active_runs(tenant: str) -> list[dict]:
    """Current runs for a tenant, newest first, without exposing their full logs."""
    active = [job for job in _JOBS.values()
              if job["tenant"] == tenant and job["status"] == "running"]
    active.sort(key=lambda job: job["started_at"], reverse=True)
    return [status(job["run_id"]) for job in active]


def pending_runs(tenant: str) -> list[dict]:
    """Running and successfully completed, still-unpublished runs for a tenant.

    Completed jobs stay in the in-process registry after the wizard unmounts. Exposing
    them here lets the model catalogue distinguish "training finished" from "published"
    and offer the user a route back to Review & Publish.
    """
    pending = [job for job in _JOBS.values()
               if job["tenant"] == tenant
               and job["status"] in ("running", "done")
               and not job.get("published", False)]
    pending.sort(key=lambda job: job["started_at"], reverse=True)
    return [status(job["run_id"]) for job in pending]


def mark_published(tenant: str, model_key: str, version: str) -> None:
    """Remove the matching completed candidate from the pending catalogue state."""
    matching_started_at = next((
        job["started_at"] for job in _JOBS.values()
        if job["tenant"] == tenant and job["model_key"] == model_key
        and str((job.get("result") or {}).get("version")) == str(version)
    ), None)
    if matching_started_at is None:
        return
    for job in _JOBS.values():
        if (job["tenant"] == tenant and job["model_key"] == model_key
                and job["started_at"] <= matching_started_at):
            job["published"] = True
