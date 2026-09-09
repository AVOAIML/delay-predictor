"""maxxflow CLI entrypoint. ``maxxflow <command> [opts]``.

Commands (each == one AML job in Phase 2):
  db-provision  create public + tenant schema (DDL from schema_def)        == infra/seed
  seed          synthetic ground-truth into tenant Postgres                == ADF ingest
  validate      run the 3 synthetic CI gates (schema/realism/leakage)      == GE validate
  fe            bronze->silver->gold features on the lake                   == AML feature step
  train         train + isotonic calibrate + register to MLflow            == AML training job
  train-csv     train a CSV model card from a raw export (local or lake)     == AML retraining job
  train-db      train a model card from the tenant Postgres schema           == AML retraining job
  score         write advisory scores back to Postgres                     == AML batch/endpoint
  drift         Evidently report                                           == AML model monitor
  serve         run the BYOC scoring script under azmlinfsrv               == AML online endpoint
  parity-stub   trivial train/score round-trip (checkpoint 0 proof)
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys

MODULES = ["m1_quote", "m2_inventory", "m3_delay", "m4_bom"]


def _module_fn(module: str, fn: str):
    if module not in MODULES:
        raise SystemExit(f"unknown module {module!r}; choose from {MODULES}")
    mod = importlib.import_module(module)
    if not hasattr(mod, fn):
        raise SystemExit(f"module {module} does not implement {fn}() yet")
    return getattr(mod, fn)


def _each_module(arg: str) -> list[str]:
    return MODULES if arg == "all" else [arg]


def cmd_db_provision(args):
    from maxxflow_data.provision import provision_tenant
    provision_tenant(args.tenant)
    print(f"provisioned schema for tenant={args.tenant}")


def cmd_seed(args):
    from maxxflow_synth.loader import seed
    summary = seed(tenant=args.tenant, module=args.module, seed=args.seed)
    print(json.dumps(summary, indent=2, default=str))


def cmd_validate(args):
    from maxxflow_synth.gates import run_all_gates
    report = run_all_gates(tenant=args.tenant, module=args.module, seed=args.seed)
    print(json.dumps(report, indent=2, default=str))
    if not report.get("passed", False):
        sys.exit(1)


def cmd_fe(args):
    for m in _each_module(args.module):
        _module_fn(m, "fe")(args.tenant)
        print(f"[{m}] features built")


def cmd_train(args):
    for m in _each_module(args.module):
        version = _module_fn(m, "train")(args.tenant)
        print(f"[{m}] trained + registered version={version}")


# The trainer's result carries the fitted model and the live log buffer; neither
# is JSON, and neither is what a caller reading the registry afterwards wants.
_UNSERIALISABLE_RESULT_KEYS = ("_model", "logs")
RESULT_ARTIFACT = "configurator_result.json"


def _tag_card_metadata(result: dict, *, model_key: str, source: str,
                       source_name: str, dataset_row_count: int,
                       published_at: str | None = None,
                       fallback_used: bool = False,
                       fallback_reason: str | None = None) -> None:
    """Persist model-card fields on the exact candidate/champion version."""
    from maxxflow_mlops.registry import MLflowRegistry

    MLflowRegistry().set_version_tags(
        name=result["registered_name"],
        version=str(result["version"]),
        tags={
            "data_source_kind": "database" if source == "db" else "csv",
            "data_source_name": source_name,
            "dataset_row_count": int(dataset_row_count),
            "algorithm": result.get("selected_model") or (
                "lightgbm-isotonic" if model_key == "m1_quote_line_win" else None
            ),
            "published_at": published_at,
            "fallback_used": str(fallback_used).lower(),
            "fallback_reason": fallback_reason,
        },
    )


def _publish_result_artifact(tenant: str, model_key: str, result: dict) -> None:
    """Attach the trainer's own result dict to its MLflow run as JSON.

    The in-process backend hands the UI whatever train() returned. An Azure ML run
    happens in another process on another machine, so the Configurator can only see
    what reached the registry — and metrics alone are not the same shape. Rather
    than have the UI cope with two shapes, or parse the job's stdout, the run
    carries its own result and the backend reads it back.

    Failure here is logged, never fatal: the model IS registered by this point, and
    losing a display artifact must not turn a successful train into a failed one."""
    import json as _json

    from mlflow.tracking import MlflowClient

    from maxxflow_mlops.naming import registered_model_name
    from maxxflow_mlops.registry import MLflowRegistry

    try:
        payload = {k: v for k, v in result.items() if k not in _UNSERIALISABLE_RESULT_KEYS}
        reg = MLflowRegistry()
        client = MlflowClient(reg.tracking_uri, reg.tracking_uri)
        mv = client.get_model_version(registered_model_name(tenant, model_key),
                                      str(result["version"]))
        client.log_text(mv.run_id, _json.dumps(payload, default=str), RESULT_ARTIFACT)
        print(f"attached {RESULT_ARTIFACT} to run {mv.run_id}", file=sys.stderr)
    except Exception as e:
        print(f"WARNING: could not attach {RESULT_ARTIFACT} "
              f"({type(e).__name__}: {e}) — the UI will fall back to metrics only",
              file=sys.stderr)


def _train_inventory_and_report(args, *, frame, tenant: str, source: str, logger,
                                publisher, source_name: str,
                                dataset_row_count: int) -> None:
    import contextlib
    import json as _json
    from datetime import datetime, timezone

    from m2_inventory.csv_training import publish, train_for_configurator

    fallback_used = bool(frame.attrs.get("fallback_used"))
    fallback_reason = frame.attrs.get("fallback_reason")
    if fallback_used:
        logger.log(
            "WARNING: Real snapshot history is unavailable. Training with generated "
            "bootstrap history; metrics are not real historical performance evidence."
        )

    with contextlib.redirect_stdout(sys.stderr):
        result = train_for_configurator(
            frame, tenant, register=True, source=source, logger=logger
        )
    result["fallback_used"] = fallback_used
    result["fallback_reason"] = fallback_reason
    _tag_card_metadata(
        result, model_key="m2_inventory", source=source, source_name=source_name,
        dataset_row_count=dataset_row_count, fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )
    _publish_result_artifact(tenant, "m2_inventory", result)
    out = {"model": "m2_inventory", "tenant": tenant,
           "version": result["version"], "metrics": result["metrics"],
           "source": source, "published": None,
           "fallback_used": fallback_used, "fallback_reason": fallback_reason}
    if args.publish:
        with contextlib.redirect_stdout(sys.stderr):
            decision = publish(
                tenant, str(result["version"]), result["metrics"], force=args.force
            )
        out["published"] = decision["published"]
        out["blocker"] = decision.get("blocker")
        out["gate_checks"] = decision.get("gate_checks")
        if out["published"]:
            published_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            _tag_card_metadata(
                result, model_key="m2_inventory", source=source,
                source_name=source_name, dataset_row_count=dataset_row_count,
                published_at=published_at, fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            )
            out["published_at"] = published_at
    print(_json.dumps(out, indent=2, default=str))
    if publisher is not None:
        logger.set_progress(100, "complete", "Training complete")
        message = ("FINISHED" if not args.publish or out["published"]
                   else f"NOT PUBLISHED - {out.get('blocker')}")
        publisher.append(message)
        publisher.close("FINISHED")
    if args.publish and not out["published"]:
        sys.exit(3)


def cmd_train_csv(args):
    """Train one M1 model card from a raw combined export, and optionally publish.

    THE GAP THIS FILLS. Three training paths existed and none could be scheduled:
    ``maxxflow train`` reads Postgres via the DAL, ``services/train_csv.py`` reads
    two local gold CSVs and has the price model commented out, and the path that
    actually produces the models in the registry - a raw export through
    ``csv_line_win`` and friends - existed ONLY as a background thread inside the
    Configurator API, reachable only by a human clicking Upload.

    That is why the monthly retrain had no entrypoint and why training competes
    with the API for one CPU. This command is that entrypoint: the same frame
    builders the UI uses (m1_quote.frames, imported by both rather than
    reimplemented), against a CSV that may live in the lake, runnable by an AML
    job or a Container Apps Job. Those builders live in the LIBRARY, not in
    services/: only libs/ and modules/ are installed as packages, so an earlier
    version that imported services.configurator.jobs died in the Azure ML
    training container with "No module named \'services\'".

    EXIT CODES. 0 trained (and published if asked). 3 trained but the promotion
    gate REFUSED to publish. 1 anything broke. 3 is separate on purpose: a
    scheduled retrain whose candidate is rejected has not failed - the gate did
    its job - but it must not report success either, or a month of refused
    retrains looks green in job history.
    """
    import contextlib
    import json as _json
    import os

    import pandas as pd

    # BEFORE get_settings(), which is lru_cached and reads the environment once.
    # Azure ML injects its own MLFLOW_TRACKING_URI into every job container and it
    # won over the one we set in the job spec, so the first AML run registered
    # nowhere and died inside MlflowClient. An explicit flag is the only form the
    # platform cannot override.
    _redirect_registry(args.mlflow_uri)
    # AML also pre-seeds the RUN CONTEXT, not just the tracking URI: it exports
    # MLFLOW_RUN_ID (and friends) pointing at the AML run it created for this job,
    # so that mlflow.start_run() attaches to it instead of opening a new one. Once
    # we redirect the tracking URI to our own server those ids refer to runs that
    # exist only in AML's store, and start_run() fails with
    #   RESOURCE_DOES_NOT_EXIST: Run with id=<aml-run-name> not found
    # Clearing them makes start_run() do what it does everywhere else: create a
    # run in whichever store we are actually pointed at. The AML run still records
    # the job itself — this only detaches OUR experiment tracking from it.
    from maxxflow_core.settings import get_settings

    get_settings.cache_clear()
    from m1_quote import frames
    from m1_quote.csv_common import RunLogger

    logger, publisher = _progress_logger(args.progress_id)
    settings = get_settings()
    # storage_options ONLY for a remote URI. pandas raises "storage_options passed
    # with file object or non-fsspec file path" if they are supplied for a plain
    # local path, and lake_storage_options is populated whenever the configured
    # lake is MinIO/S3 - so passing them unconditionally breaks every local run.
    # For abfss:// no options are needed: adlfs picks up the ambient credential,
    # which on Container Apps and AML compute is the managed identity selected by
    # AZURE_CLIENT_ID. Nothing secret ever reaches the command line.
    remote = "://" in args.csv and not args.csv.startswith("file://")
    # Say which registry this run will write to. When it turns out to be the wrong
    # one, this line in the job log is the difference between a five-minute fix and
    # an afternoon.
    print(f"model registry: {settings.mlflow_tracking_uri or '(default)'}", file=sys.stderr)
    opts = (settings.lake_storage_options or None) if remote else None
    print(f"reading {args.csv}" + (" (remote)" if remote else " (local)"), file=sys.stderr)
    raw = pd.read_csv(args.csv, storage_options=opts)
    print(f"{len(raw)} raw rows", file=sys.stderr)
    source_name = os.environ.get("MAXXFLOW_DATA_SOURCE_NAME") or args.csv.rsplit("/", 1)[-1]
    dataset_row_count = int(os.environ.get("MAXXFLOW_DATASET_ROW_COUNT") or len(raw))

    if args.model == "m2_inventory":
        _train_inventory_and_report(
            args, frame=raw, tenant=args.tenant, source="csv", logger=logger,
            publisher=publisher, source_name=source_name,
            dataset_row_count=dataset_row_count,
        )
        return

    # STDOUT IS THE MACHINE INTERFACE. MLflow prints its own run/experiment URLs
    # to stdout on run end, so without this `maxxflow train-csv ... | jq` fails on
    # two banner lines before the document. Everything the training path prints
    # goes to stderr; stdout carries the result JSON and nothing else. A job
    # runner reads one, a human reads the other.
    with contextlib.redirect_stdout(sys.stderr):
        frames.write_option_graph(raw, logger)
        frame = frames.build_training_frame(args.model, raw, args.tenant, logger)

    # Shared with train-db: both must emit the same JSON and the same exit codes,
    # because the Configurator reads a run back through the registry without
    # knowing which command produced it.
    _train_and_report(args, frame=frame, tenant=args.tenant, source="csv", logger=logger,
                      publisher=publisher, source_name=source_name,
                      dataset_row_count=dataset_row_count)


def _redirect_registry(mlflow_uri: str) -> None:
    """Point this process at OUR registry, in the one form Azure ML cannot override.

    AML injects its own MLFLOW_TRACKING_URI into every job container and it WON
    over the value set in the job spec, so the first real run tried to register in
    the workspace registry — which has deprecated stages, not the aliases publish()
    moves — and died inside MlflowClient.

    AML also pre-seeds the RUN CONTEXT: MLFLOW_RUN_ID and friends point at the AML
    run it created for this job, so mlflow.start_run() attaches to it rather than
    opening its own. Once the tracking URI is redirected those ids name runs that
    exist only in AML's store, and start_run() fails with
      RESOURCE_DOES_NOT_EXIST: Run with id=<aml-run-name> not found
    Clearing them makes start_run() behave as it does everywhere else. The AML run
    still records the job; this only detaches our experiment tracking from it.

    Must run BEFORE get_settings(), which is lru_cached and reads the environment
    once.
    """
    import os
    if not mlflow_uri:
        return
    # Both names: MAXXFLOW_MLFLOW_URI is what _resolve_uri trusts, and
    # MLFLOW_TRACKING_URI is what any bare mlflow.* call in a library reads.
    os.environ["MAXXFLOW_MLFLOW_URI"] = mlflow_uri
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
    for stale in ("MLFLOW_RUN_ID", "MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME",
                  "MLFLOW_TRACKING_TOKEN", "MLFLOW_TRACKING_AUTH"):
        os.environ.pop(stale, None)


def _say(msg: str, logger=None) -> None:
    """Job-log line that the UI should see too.

    stderr alone lands only in the cluster's std_log.txt, which the Configurator
    fetches on FAILURE. Lines that explain a SUCCESSFUL run -- which schema, how
    many rows -- have to go through the logger to reach the Train page.
    """
    print(msg, file=sys.stderr)
    if logger is not None:
        logger.log(msg)


def _progress_logger(progress_id: str):
    """A RunLogger that also mirrors each line to MLflow, when asked to.

    Returns ``(logger, publisher | None)``. Without --progress-id this is exactly
    the RunLogger every other caller gets, so a hand-run CLI invocation pays
    nothing and behaves identically. The Configurator's AML backend mints the id
    before submitting; nothing else passes one.
    """
    from m1_quote.csv_common import RunLogger

    if not progress_id:
        return RunLogger(), None

    from maxxflow_mlops.progress import ProgressPublisher

    publisher = ProgressPublisher(progress_id)

    class _MirroredRunLogger(RunLogger):
        def log(self, msg: str) -> str:
            line = super().log(msg)
            publisher.append(line)
            return line

        def set_progress(self, percent, phase, label, *, current=None, total=None,
                         message=None):
            snapshot = super().set_progress(
                percent, phase, label, current=current, total=total, message=message
            )
            publisher.update_progress(snapshot)
            return snapshot

    return _MirroredRunLogger(), publisher


def _train_and_report(args, *, frame, tenant: str, source: str, logger, publisher=None,
                      source_name: str, dataset_row_count: int) -> None:
    """Shared tail of train-csv and train-db: fit, register, attach the result
    artifact, optionally publish, emit the result JSON, and set the exit code.

    Both commands must produce the SAME stdout document and the SAME exit codes,
    because the Configurator reads a run back through the registry without knowing
    which one produced it. Keeping this in one function is what guarantees that;
    two copies would drift the moment either grew a field.
    """
    import contextlib
    import json as _json

    trainer_mod = {"m1_quote_win": "csv_win", "m1_quote_price": "csv_price",
                   "m1_quote_line_win": "csv_line_win", "m1_quote_mil": "csv_mil"}[args.model]
    trainer = importlib.import_module(f"m1_quote.{trainer_mod}")

    with contextlib.redirect_stdout(sys.stderr):
        kwargs = {"logger": logger, "register": True, "source": source}
        if args.model != "m1_quote_mil":
            kwargs["auto_hpo"] = not args.no_hpo
        result = trainer.train(frame, tenant, **kwargs)
        for line in logger.lines:
            print(line)

    _tag_card_metadata(
        result, model_key=args.model, source=source, source_name=source_name,
        dataset_row_count=dataset_row_count,
    )
    _publish_result_artifact(tenant, args.model, result)

    out = {"model": args.model, "tenant": tenant, "version": result["version"],
           "metrics": result["metrics"], "source": source, "published": None}
    if args.publish:
        with contextlib.redirect_stdout(sys.stderr):
            decision = trainer.publish(tenant, str(result["version"]), result["metrics"],
                                       force=args.force)
        out["published"] = decision["published"]
        out["blocker"] = decision.get("blocker")
        out["gate_checks"] = decision.get("gate_checks")
        if out["published"]:
            from datetime import datetime, timezone
            published_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            _tag_card_metadata(
                result, model_key=args.model, source=source, source_name=source_name,
                dataset_row_count=dataset_row_count, published_at=published_at,
            )
            out["published_at"] = published_at
    print(_json.dumps(out, indent=2, default=str))
    if args.publish and not out["published"]:
        # Exit 3, not 1: the model trained and the gate refused it. The feed is
        # closed FINISHED for the same reason — the run did what it was asked to.
        if publisher is not None:
            logger.set_progress(100, "complete", "Training complete")
            publisher.append(f"NOT PUBLISHED - {out['blocker']}")
            publisher.close("FINISHED")
        print(f"NOT PUBLISHED - {out['blocker']}", file=sys.stderr)
        sys.exit(3)
    if publisher is not None:
        logger.set_progress(100, "complete", "Training complete")
        publisher.close("FINISHED")


def cmd_train_db(args):
    """Train one M1 model card from the tenant Postgres schema, on whatever compute
    runs this process — which is the point.

    THE GAP THIS FILLS. train-csv gave the CSV path an entrypoint an Azure ML job
    could invoke; the DB path had none, so source="db" was pinned to a background
    thread inside ca-configurator with the comment "an AML job would need its own
    route to that database". This is that route.

    WHAT AN AML JOB NEEDS THAT THE API DOES NOT
      * Its own network path. Serverless compute has dynamic egress addresses, so
        a single-IP firewall rule will not cover it; the server needs the
        allow-Azure-services rule (infra/aml/db-access.sh grant does both).
      * Its own credential. The API reads PGPASS from a Container Apps secretRef,
        which does not exist on AML compute. So the password is fetched HERE, at
        runtime, from Key Vault using the compute's managed identity — nothing
        secret is in the job definition, and rotating the secret needs no redeploy.

    TENANT vs DATA-TENANT. --tenant names the registered model; --data-tenant is
    the schema the rows come from. They differ for the shared base model, which is
    owned by `global` and has no schema of its own (see the Configurator's
    _data_tenant). Defaults to --tenant so a tenant-owned model needs only one flag.
    """
    import os

    from maxxflow_core.keyvault import hydrate_pg_password

    _redirect_registry(args.mlflow_uri)
    hydrate_pg_password()

    from maxxflow_core.settings import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    data_tenant = args.data_tenant or args.tenant

    from m1_quote.csv_common import RunLogger
    from m1_quote import db_features
    from m2_inventory.db_training import build_training_frame as build_inventory_frame

    builders = {"m1_quote_win": db_features.build_win_frame,
                "m1_quote_price": db_features.build_price_frame,
                "m1_quote_line_win": db_features.build_line_frame,
                "m2_inventory": build_inventory_frame}
    if args.model not in builders:
        print(f"{args.model} has no DB-path builder; it trains from an uploaded CSV. "
              f"Models with a DB path: {', '.join(sorted(builders))}.", file=sys.stderr)
        sys.exit(1)

    print(f"model registry: {settings.mlflow_tracking_uri or '(default)'}", file=sys.stderr)
    # data_db_url_safe, never data_db_url: the latter carries the password and this
    # line lands in a job log that outlives the run.
    print(f"database: {settings.data_db_url_safe or '(not configured)'}", file=sys.stderr)
    if not settings.db_enabled:
        print("no database configured — set PGHOST/PGUSER/PGDATABASE (and PGPASS, or "
              "KEYVAULT_NAME to fetch it).", file=sys.stderr)
        sys.exit(1)
    # Say it HERE, not thirty frames deep in a driver. Without a password psycopg
    # fails with "fe_sendauth: no password supplied" under a SQLAlchemy pool
    # traceback, which names neither the variable that is missing nor the fetch
    # that was supposed to supply it. The most likely cause by far is that
    # KEYVAULT_NAME never reached the job, so name that first.
    # pg_host set + no password anywhere = the URL was assembled from PARTS and the
    # secret step did not happen. An explicit DATA_DB_URL embeds its own password
    # and leaves PGPASS empty, which is fine and must not trip this.
    if settings.pg_host and not settings.data_db_has_password:
        print("no Postgres password. PGPASS is unset and nothing was fetched from Key "
              f"Vault (KEYVAULT_NAME={os.environ.get('KEYVAULT_NAME') or 'unset'}). "
              "For an Azure ML job the Configurator passes KEYVAULT_NAME through, so "
              "set it on the app:\n"
              "  az containerapp update -n ca-configurator -g $RG --set-env-vars "
              "KEYVAULT_NAME=<vault> PG_PASSWORD_SECRET=pg-admin-password",
              file=sys.stderr)
        sys.exit(1)
    logger, publisher = _progress_logger(args.progress_id)
    # Into the feed as well as the log: "reading schema tenant_demo" is the line
    # that tells you the data tenant resolved correctly, and it is written before
    # the trainer exists, so nothing else would carry it to the UI.
    _say(f"reading schema {settings.tenant_schema(data_tenant)}", logger)
    import contextlib
    with contextlib.redirect_stdout(sys.stderr):
        frame = builders[args.model](data_tenant)
    _say(f"{len(frame)} training rows x {len(frame.columns)} columns", logger)
    if frame.empty:
        msg = ("inventory_ml_snapshots returned no rows."
               if args.model == "m2_inventory" else
               "the query returned no rows — the source tables may be empty, or every "
               "quotation's stage/status resolved to neither won nor lost.")
        _say(msg, logger)
        if publisher is not None:
            publisher.close("FAILED")
        sys.exit(1)

    report = _train_inventory_and_report if args.model == "m2_inventory" else _train_and_report
    report(args, frame=frame, tenant=args.tenant, source="db", logger=logger,
           publisher=publisher,
           source_name=frame.attrs.get("source_name") or
                       os.environ.get("MAXXFLOW_DATA_SOURCE_NAME") or "MaXXflow Database",
           dataset_row_count=len(frame))


def cmd_score(args):
    for m in _each_module(args.module):
        n = _module_fn(m, "score")(args.tenant)
        print(f"[{m}] scored {n} rows -> Postgres")


def cmd_drift(args):
    for m in _each_module(args.module):
        path = _module_fn(m, "drift")(args.tenant)
        print(f"[{m}] drift report -> {path}")


def cmd_serve(args):
    # Run the exact BYOC scoring script under azmlinfsrv (same as the AML endpoint).
    import os
    import subprocess
    score_path = os.path.join(os.path.dirname(__file__), "..", "..", "docker", "score.py")
    score_path = os.path.abspath(score_path)
    try:
        subprocess.run(["azmlinfsrv", "--entry_script", score_path, "--port", str(args.port)], check=True)
    except FileNotFoundError:
        raise SystemExit(
            "azmlinfsrv not found. Install the serve extra (`uv pip install -e '.[serve]'`) "
            "or use the docker model-server: `docker compose -f compose.local.yml up -d model-server`."
        )


def cmd_parity_stub(args):
    from maxxflow_mlops.parity_stub import FEATURES, train_parity_stub
    from maxxflow_mlops.serving import ModelRouter
    if args.action == "train":
        v = train_parity_stub(tenant=args.tenant)
        print(f"parity-stub trained + champion set, version={v}")
    else:
        router = ModelRouter()
        recs = [{"x0": 1.5, "x1": -0.3}, {"x0": -2.0, "x1": 0.4}]
        out = router.predict(tenant=args.tenant, module="parity", records=recs)
        print(json.dumps(out, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="maxxflow", description="MaXXFlow EPIC 10 AI/ML — local twins of the AML jobs")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp, with_module=True):
        sp.add_argument("--tenant", default="demo")
        if with_module:
            sp.add_argument("--module", default="all")
        sp.add_argument("--seed", type=int, default=7)

    add_common(sub.add_parser("db-provision"), with_module=False)
    add_common(sub.add_parser("seed"))
    add_common(sub.add_parser("validate"))
    add_common(sub.add_parser("fe"))
    add_common(sub.add_parser("train"))
    sp_tc = sub.add_parser("train-csv")
    sp_tc.add_argument("--model", required=True,
                       choices=["m1_quote_win", "m1_quote_price", "m1_quote_line_win",
                                "m1_quote_mil", "m2_inventory"])
    sp_tc.add_argument("--csv", required=True,
                       help="raw combined export: a local path or a lake URI "
                            "(abfss://... / s3://...)")
    sp_tc.add_argument("--tenant", default="global",
                       help="names the registered model only; never filters rows")
    sp_tc.add_argument("--publish", action="store_true",
                       help="move @champion through the promotion gate if it passes")
    sp_tc.add_argument("--force", action="store_true",
                       help="override failed model-performance gates")
    sp_tc.add_argument("--no-hpo", action="store_true")
    sp_tc.add_argument("--progress-id", default="",
                       help="correlation id for the live progress feed. The Configurator mints one before submitting so the UI can tail this run; omit it and nothing is published.")
    sp_tc.add_argument("--mlflow-uri", default="",
                       help="registry to write to. Overrides MLFLOW_TRACKING_URI, which "
                            "Azure ML injects into every job and which therefore cannot "
                            "be trusted inside one.")

    # train-db mirrors train-csv's contract deliberately: same --publish/--force/
    # --no-hpo semantics, same stdout JSON, same exit code 3 for "trained but the
    # gate refused". A caller should not need to know which source produced a run.
    sp_td = sub.add_parser("train-db")
    sp_td.add_argument("--model", required=True,
                       choices=["m1_quote_win", "m1_quote_price", "m1_quote_line_win",
                                "m2_inventory"])
    sp_td.add_argument("--tenant", default="global",
                       help="names the registered model")
    sp_td.add_argument("--data-tenant", default="",
                       help="schema the rows come from. Defaults to --tenant; differs "
                            "for the shared base model, which owns no schema.")
    sp_td.add_argument("--publish", action="store_true",
                       help="move @champion through the promotion gate if it passes")
    sp_td.add_argument("--force", action="store_true",
                       help="override failed model-performance gates")
    sp_td.add_argument("--no-hpo", action="store_true")
    sp_td.add_argument("--progress-id", default="",
                       help="correlation id for the live progress feed. The Configurator mints one before submitting so the UI can tail this run; omit it and nothing is published.")
    sp_td.add_argument("--mlflow-uri", default="",
                       help="registry to write to. Overrides MLFLOW_TRACKING_URI, which "
                            "Azure ML injects into every job and which therefore cannot "
                            "be trusted inside one.")

    add_common(sub.add_parser("score"))
    add_common(sub.add_parser("drift"))

    sp_serve = sub.add_parser("serve")
    sp_serve.add_argument("--port", type=int, default=5001)

    sp_ps = sub.add_parser("parity-stub")
    sp_ps.add_argument("action", choices=["train", "score"])
    sp_ps.add_argument("--tenant", default="demo")

    return p


_DISPATCH = {
    "db-provision": cmd_db_provision,
    "seed": cmd_seed,
    "validate": cmd_validate,
    "fe": cmd_fe,
    "train": cmd_train,
    "train-csv": cmd_train_csv,
    "train-db": cmd_train_db,
    "score": cmd_score,
    "drift": cmd_drift,
    "serve": cmd_serve,
    "parity-stub": cmd_parity_stub,
}


def main(argv=None):
    args = build_parser().parse_args(argv)
    _DISPATCH[args.command](args)


if __name__ == "__main__":
    main()
