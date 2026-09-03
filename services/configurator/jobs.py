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
                "m1_quote_line_win": db_features.build_line_frame}


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
          auto_hpo: bool = True, data_tenant: str | None = None) -> str:
    """`tenant` owns the model (it becomes the registered name). `data_tenant` is
    whose schema the DB path reads. They are the same for a tenant-owned model and
    differ for the shared base model, which is owned by `global` and has no schema
    of its own. Defaults to tenant, so the CSV path is unaffected."""
    if model_key not in _TRAINERS:
        raise KeyError(model_key)
    run_id = uuid.uuid4().hex[:12]
    read_from = data_tenant or tenant
    logger = RunLogger()
    job = {"run_id": run_id, "tenant": tenant, "model_key": model_key, "source": source,
           "data_tenant": read_from,
           "status": "running", "logger": logger, "result": None, "error": None,
           "started_at": time.time()}
    _JOBS[run_id] = job

    def _run():
        try:
            if model_key == "m2_inventory":
                if source != "csv":
                    raise ValueError("m2_inventory currently retrains from CSV only")
                job["result"] = train_inventory_csv(
                    csv_path, tenant, artifact_dir=_INVENTORY_ARTIFACTS,
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
            job["status"] = "done"
            logger.log("Training complete — candidate registered. Review results, then Publish.")
        except Exception as e:
            job["error"] = f"{type(e).__name__}: {e}"
            job["status"] = "error"
            logger.log(f"ERROR: {job['error']}")

    threading.Thread(target=_run, daemon=True).start()
    return run_id


def status(run_id: str) -> dict:
    job = _JOBS.get(run_id)
    if job is None:
        return {"status": "unknown"}
    out = {"run_id": run_id, "tenant": job["tenant"], "model_key": job["model_key"],
           "source": job["source"], "status": job["status"], "logs": list(job["logger"].lines),
           "elapsed_s": round(time.time() - job["started_at"], 1)}
    if job["status"] == "done":
        r = dict(job["result"]); r.pop("_model", None); r.pop("logs", None)
        out["result"] = r
    if job["status"] == "error":
        out["error"] = job["error"]
    return out
