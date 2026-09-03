# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MaXXFlow EPIC 10 — classical-ML advisory platform for the MaXXFlow multi-tenant MRP/ERP. Four
modules (M1 Smart Quote Optimiser, M2 Predictive Inventory, M3 Production Delay, M4 BOM Cleanup;
M5 Job Scheduling is a `NotImplementedError` stub) produce advisory scores written back into the
product's existing `customElements` JSONB columns — never a blocking model call on page load.
**Classical ML only** (LightGBM / scikit-learn / statsforecast / numpy-scipy): no transformer is
hosted or trained here; LLM/embedding calls are external APIs behind provider ports with
deterministic local stubs (tests never call a paid API).

The whole system is built **ports-and-adapters** so the same module code runs locally today and on
Azure in Phase 2, selected purely by a config profile (`APP_ENV`) — there is no `if env == ...`
branching in module code, and this is enforced by `tests/parity`. Only the local adapters are
implemented; Azure (Bicep infra, AML pipelines, ADF/Event Grid/Service Bus) is Phase 2 and not yet
authored — see `NOTES.md` for the go-live checklist and credential/decision list.

## Commands

```bash
# environment (uv is the only dependency manager; single pinned env baked into the ACR image)
uv sync --extra dev --extra serve --extra ui

# one-time stack bring-up + tenant schema
make bootstrap TENANT=demo

# per-module inner loop (each target == one future Azure ML pipeline step)
make seed     TENANT=demo MODULE=m1_quote   # synthetic ground truth -> tenant Postgres
make validate TENANT=demo MODULE=m1_quote   # 3 CI gates: schema / realism / leakage+learnability
make fe       TENANT=demo MODULE=m1_quote   # bronze -> silver -> gold features on MinIO
make train    TENANT=demo MODULE=m1_quote   # LightGBM + isotonic -> register to MLflow
make serve                                   # azmlinfsrv model-server on :5001
make score    TENANT=demo MODULE=m1_quote   # advisory writeback to <entity>.customElements
make drift    TENANT=demo MODULE=m1_quote   # Evidently/PSI report -> reports/
# MODULE=all runs every module; the `maxxflow` CLI backs every target directly:
#   maxxflow seed|validate|fe|train|score|drift|serve --tenant <t> --module <m>

# human-facing surfaces
make ui                 # Configurator API :8000, dashboard :8501, React UI :5173
make start               # build + start the entire compose stack at once

# M1 CSV/DB training path (works without live Postgres data)
make train-csv TENANT=furniched
python services/train_csv.py --publish                        # trains+publishes the shared `global` base model
python services/train_csv.py --tenant furniched --publish     # a tenant trains+publishes its own

# tests
make test                                          # full suite (~1-2 min)
uv run --extra dev pytest tests/unit -q            # one group (unit/contract/data_quality/smoke/parity)
uv run --extra dev pytest tests/unit/test_m3_leakage.py -q            # one file
uv run --extra dev pytest tests/unit/test_m3_leakage.py::test_name -q # one test

# frontend (Configurator React UI)
cd frontend && npm install
VITE_API_BASE=http://localhost:8000 npm run dev    # standalone hot-reload dev server, :5173
cd frontend && npm run build                        # tsc -b && vite build

make clean               # compose down -v + wipe lake/reports/mlflow.db/mlruns/artifacts
```

There is no configured linter/formatter (no ruff/black/mypy in `pyproject.toml`) — don't invent a
lint step that isn't there.

Test markers (`pyproject.toml`): `parity`, `data_quality`, `smoke`, `contract`, `needs_db` (skips
automatically when no live Postgres is reachable via `DATA_DB_URL`, so `make test` is green both
before and after `make bootstrap`). `tests/conftest.py` runs everything offline by default — MLflow
against a temp sqlite registry, the feature lake against a temp filesystem, providers against
stubs — and puts `libs/` + `modules/` on `sys.path` without needing an editable install.

macOS: LightGBM needs `brew install libomp`. Python must be 3.10–3.12 (numpy has no cp313 wheels;
`.python-version` pins 3.12). MLflow UI is at `:8085`, not `:5000` (macOS AirPlay owns `:5000`).

## Architecture

**Data flow (see `docs/WORKFLOW_COMPONENT_REFERENCE.md` for full What/Why/How per node):**

```
sources (synth | live tenant MRP | CSV upload)
  -> DAL (maxxflow_data, search_path tenant isolation) + transforms (Decimal, as-of clock, HMAC tiers)
  -> 3 CI gates (schema, realism, leakage+learnability — AUC≈1.0 = FAIL)
  -> medallion lake (fsspec: bronze -> silver -> gold, MinIO==ADLS)
  -> training (LightGBM + isotonic/quantile/combiner + light auto-HPO)
  -> MLflow registry, name = t_<tenant>__m_<module>, @champion/@previous alias
  -> publish gate (candidate must beat champion on AUC & Brier & ECE, or price coverage & pinball)
  -> serving (BYOC azmlinfsrv, ONE endpoint routes by name@champion; M2 is batch)
  -> writeback (<entity>.customElements JSON + audit_logs for suppressed/low-confidence outputs)
  -> monitoring (Evidently / PSI drift -> reports/)
```

**`libs/` — shared code, one distribution:**
- `maxxflow_core` — settings (typed `Settings`, never read `APP_ENV` directly in module code),
  the single TZ-explicit as-of clock, `decimal.Decimal` money math, HMAC hashing, MasterData
  constants, and `ports.py` (the `DataSource`/`LakeIO`/`ModelRegistry`/`LLMProvider`/
  `EmbeddingProvider`/`EventQueue`/`DriftReporter` interfaces every module depends on instead of a
  concrete backend).
- `maxxflow_data` — the DAL: `engine.py` owns `search_path` tenant isolation and a read-guard that
  rejects any SQL containing `tenant_id` or the stale `rop_status`; `transforms.py` holds the
  parity-critical conversions (ROP derive, GRN on-time off status-transition timestamp not
  `createdAt`, operator HMAC->skill-tier, M3 as-of-T censoring, Bayesian-smoothed win-rates).
- `maxxflow_synth` — synthetic generator (NumPy simulators with a hidden latent ground-truth
  function) plus the 3 CI validation gates in `gates.py` (per-module AUC/positive-rate bands).
- `maxxflow_features` — medallion lake IO via fsspec/pyarrow (only the URI scheme, `s3://` vs
  `abfss://`, changes between local and Azure).
- `maxxflow_events` — in-process retrain queue + debounced micro-batch worker; the local stand-in
  for Event Grid/Service Bus. The event envelope carries the authoritative payload + label
  (replica-lag discipline) for M3/M4.
- `maxxflow_mlops` — `registry.py` (name+`@champion` only, no search/stage API — those fail
  silently on Azure ML), `promotion.py` (the champion/challenger gate), `serving.py` (the BYOC
  `ModelRouter`, LRU-caches loaded models, asserts the tenant tag after load), `drift.py`.
- `maxxflow_providers` — LLM/embeddings behind the port interfaces: deterministic stub locally,
  Azure OpenAI/Foundry seam for Phase 2 (currently raises `NotImplementedError` until wired).
- `maxxflow_cli` — the `maxxflow` command; each subcommand maps 1:1 to a `make` target and a
  future Azure ML pipeline job.

**`modules/` — one uniform vertical slice per module** (`m1_quote` is the reference
implementation): `dal.py` (read replica) -> `features.py`/`db_features.py` (adaptive — uses
whichever agreed columns are present and skips absent ones without synthesizing) -> `train.py` ->
`model.py` (MLflow pyfunc wrapper bundling booster + calibrator/quantile-regressors + category
vocab + feature list) -> `score.py` (advisory writeback) -> `pipeline.py` (orchestration) ->
`synth.py` (synthetic generator for that module). M1 additionally has `csv_common.py`/`csv_win.py`/
`csv_price.py`/`raw_ingest.py`/`live_features.py` for the CSV/DB Configurator training path
(`dataset/gold_quote_win.csv`, `dataset/gold_price_band.csv`).

**Multi-tenant routing:** models register as `t_<tenant>__m_<module>`; the name *is* the routing
key. A shared `global` tenant holds base models trained on pooled data — a newly onboarded tenant
with no model of its own is served the `global` base until it trains and publishes its own
champion. Training is tenant-agnostic: `tenant` only names the registered model, never a feature,
never a row filter.

**`services/`** — `configurator/app.py` (FastAPI :8000, the UI's only backend: model cards,
predict, 5-step retrain wizard, background train with live logs, publish/champion-gate),
`dashboard/app.py` (Streamlit :8501, read-only overview reading the MLflow registry + model-server
`/health` + `reports/`), `train_csv.py` (CLI entry for the M1 CSV training path incl. `--publish`).

**`frontend/`** — React + Vite + TypeScript Configurator UI (`src/App.tsx` model cards,
`TestPredictions.tsx` what-if drawer, `RetrainWizard.tsx` 5-step wizard + publish modal,
`api.ts`/`types.ts` typed client), talks to the Configurator API via `VITE_API_BASE` (CORS-open).

**Config:** `APP_ENV` selects a profile in `config/profiles/` (`local` is fully populated; other
profiles are Phase-2 seams with `TODO(...)` markers — verify before building, don't guess).
`.env.local` (git-ignored) layers machine-specific secrets last. Module code always reads typed
`Settings` fields, never `APP_ENV` directly.

## Non-negotiables (enforced by tests, not just documented)

- **Tenant isolation via `search_path`**, never `WHERE tenant_id` — a read guard rejects SQL
  containing `tenant_id` or the stale `rop_status`.
- **Classical ML only** — no transformer hosted/trained; LLM/embeddings are external ports with
  deterministic stubs, never a paid API in tests.
- **`decimal.Decimal`** for all margin / ±30% price clamp / 110% overrun math — no float drift.
- **One TZ-explicit as-of clock** — identical features regardless of process timezone.
- **PII never leaves the DAL** — operator UUIDs are HMAC'd to a skill-tier before bronze.
- **Derive, don't read** — ROP state is derived live; GRN on-time keys off the status-transition
  timestamp, not `createdAt`.
- **No leakage** — M3 features are censored as-of-T; the data-quality gate fails any model that
  looks *too good* (AUC≈1.0) because that means a label leaked.
- **Registry: name + `@champion` only** — no search/stage APIs (they fail silently on Azure ML).
- **BYOC serving** — `azmlinfsrv` locally and on Azure, through one scoring script (`docker/score.py`).

`NOTES.md` §5 maps every one of these rules to the exact file/test that enforces it — check there
before assuming where a guardrail lives.

## Documentation map

- `docs/WORKFLOW_COMPONENT_REFERENCE.md` — every architecture component, What/Why/How.
- `docs/CODEBASE_WALKTHROUGH.md` — per-file, per-function reference for the whole repo.
- `docs/M1_TRAIN_INFERENCE_MAP.md` — M1 train↔inference field/feature mapping.
- `NOTES.md` — assumptions baked into the build, what's verified vs. unrun, and the Phase-2
  (Azure) go-live checklist with exact credential/decision points.
- `db/README.md` — how the tenant-template DDL is generated.
