<div align="center">

# MaXXFlow EPIC 10 — AI/ML Platform

**Classical-ML advisory intelligence for the MaXXFlow multi-tenant MRP/ERP.**
One codebase, environment-swapped by config — local today, Azure in Phase 2, with no module rewrites.

![Python](https://img.shields.io/badge/python-3.10--3.12-3776AB?logo=python&logoColor=white)
![ML](https://img.shields.io/badge/ML-LightGBM%20%7C%20scikit--learn-F7931E)
![Registry](https://img.shields.io/badge/registry-MLflow-0194E2?logo=mlflow&logoColor=white)
![Serving](https://img.shields.io/badge/serving-azmlinfsrv%20(BYOC)-0078D4?logo=microsoftazure&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![Status](https://img.shields.io/badge/status-local%20preview-yellow)
![License](https://img.shields.io/badge/license-Proprietary-red)

</div>

---

## Table of contents

- [Overview](#overview)
- [Why classical ML](#why-classical-ml)
- [The four modules](#the-four-modules)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Repository layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [The inner loop](#the-inner-loop)
- [Smart Quote Optimiser (M1) — UI & training](#smart-quote-optimiser-m1--ui--training)
- [Multi-tenant model routing & the `global` base model](#multi-tenant-model-routing--the-global-base-model)
- [Configuration](#configuration)
- [Service ports](#service-ports)
- [Testing](#testing)
- [Design principles (non-negotiables)](#design-principles-non-negotiables)
- [Local ⇄ Azure parity](#local--azure-parity)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

MaXXFlow EPIC 10 adds four AI/ML capabilities to the MaXXFlow MRP/ERP, each surfaced as an
**advisory** signal written back to the product's existing data model (the app renders a cached
score — no blocking model call on page load). The platform is **classical-ML only** (LightGBM /
scikit-learn / statsforecast / numpy-scipy): **no transformer model is hosted or trained here.**
LLM and embedding calls are external APIs behind provider interfaces, with deterministic stubs
locally, so tests never call a paid API.

The system is multi-tenant (schema-per-tenant Postgres), tracks and serves models through MLflow,
and is built ports-and-adapters style so the **same code** runs on a laptop today and on Azure in
Phase 2, selected purely by a configuration profile — there is no `if env == ...` branching anywhere
in module code (enforced by a CI parity test).

> **Scope: LOCAL.** Everything runs on the `compose.local.yml` stack (Postgres + MinIO + MLflow +
> model-server + UI). Azure (Bicep infra, AML pipelines, ADF/Event Grid/Service Bus) is Phase 2 —
> the parity *seams* are built now; only the local adapters are implemented. See
> [`NOTES.md`](NOTES.md) for the go-live checklist.

## Why classical ML

The workloads are tabular, need calibrated probabilities and hard latency/cost budgets, and must be
explainable and reproducible per tenant. Gradient-boosted trees (LightGBM) with isotonic calibration
beat deep models here on cost, speed and honesty of output. Text similarity for BOM cleanup uses
TF-IDF character n-grams + RapidFuzz rather than embeddings. The result runs comfortably on a modest
CPU and produces auditable, calibrated advisories.

## The four modules

| Module | Predicts | Model | Retrain cadence | Serving → writeback |
|---|---|---|---|---|
| **M1 — Smart Quote Optimiser** | Calibrated win probability + recommended price band | LightGBM classifier + isotonic; 3 LightGBM quantile regressors (P25/P50/P75) vs empirical baseline | Monthly / on-demand | Online endpoint → `Quotation.customElements` |
| **M2 — Predictive Inventory** | 30/60-day stockout risk | LightGBM + isotonic; EWMA/Croston demand | Weekly | **Batch** → `Item.customElements` |
| **M3 — Production Delay** | P(delay) + overrun hours at the 25% milestone | Two LightGBM heads (classifier + regressor), leakage-safe as-of-T features | On MO completion (event) | Online endpoint → `ManufacturingOrder.customElements` |
| **M4 — BOM Cleanup** | Per-line error probability + suggested fix | Deterministic rules + classical anomaly (TF-IDF char n-grams + RapidFuzz) → LogisticRegression combiner | On confirmed correction (event) | Online endpoint → `BOM.customElements` |
| M5 — Job Scheduling | — | OR-Tools CP-SAT (+ deferred RL) | — | **Phase B — explicit stub only** |

Every module is a uniform **vertical slice**: `dal` (read replica) → `features` → `train` →
`model` (MLflow pyfunc wrapper) → `score` (advisory writeback) → `pipeline` (orchestration) →
`synth` (synthetic generator). M1 additionally has a CSV/DB Configurator training path.

## Architecture

```
                    ┌───────────────────────────────────────────────────────────┐
   Data sources     │  Synthetic generator  │  Live tenant MRP  │  CSV upload    │
                    └───────────────┬───────────────────────────────────────────┘
                                    ▼
   DAL (search_path isolation)   maxxflow_data ── transforms (Decimal · as-of clock · HMAC tiers)
                                    ▼
   3 CI validation gates         schema · realism · leakage+learnability  (AUC≈1.0 = FAIL)
                                    ▼
   Medallion lake (fsspec)       bronze → silver → gold  (MinIO == ADLS)
                                    ▼
   Training                      LightGBM + isotonic / quantile / combiner  + auto-HPO
                                    ▼
   MLflow registry               name = t_<tenant>__m_<module>  ·  @champion / @previous alias
                                    ▼
   Publish gate                  champion/challenger (beat AUC & Brier & ECE, or keep current)
                                    ▼
   Serving (BYOC azmlinfsrv)     ONE endpoint routes by name@champion  ·  M2 = batch
                                    ▼
   Writeback                     <entity>.customElements (JSON) + audit_logs
                                    ▼
   Monitoring                    Evidently / PSI drift → reports/
```

The design is **ports & adapters**: module code depends only on the interfaces in
`maxxflow_core/ports.py` (`DataSource`, `LakeIO`, `ModelRegistry`, `LLMProvider`,
`EmbeddingProvider`, `EventQueue`, `DriftReporter`). A config-driven factory selects the local
adapter now and the Azure adapter in Phase 2. For the full picture see
[`docs/WORKFLOW_COMPONENT_REFERENCE.md`](docs/WORKFLOW_COMPONENT_REFERENCE.md).

## Tech stack

| Concern | Technology |
|---|---|
| Language / packaging | Python 3.10–3.12, [`uv`](https://docs.astral.sh/uv/) (single locked env) |
| Models | LightGBM, scikit-learn (isotonic calibration, LogisticRegression), numpy/scipy |
| Text similarity | RapidFuzz + scikit-learn TF-IDF char n-grams (no transformers) |
| Data | PostgreSQL 16 (schema-per-tenant), SQLAlchemy + psycopg |
| Feature lake | Parquet on fsspec + pyarrow — MinIO locally, ADLS Gen2 in prod |
| Tracking & registry | MLflow (Postgres backend + MinIO/S3 artifacts) |
| Serving | `azureml-inference-server-http` (azmlinfsrv) — same BYOC image locally and on Azure ML |
| Config | pydantic-settings profiles |
| Drift | Evidently (with a dependency-free PSI fallback) |
| Data quality | pandera + custom leakage/learnability gates |
| API / UI | FastAPI (Configurator), Streamlit (dashboard), React + Vite + TypeScript (frontend) |
| Orchestration | `make` targets (each maps 1:1 to a future Azure ML job) + a `maxxflow` CLI |

## Repository layout

```
config/            pydantic-settings profiles (local / dev-azure / staging / prod)
libs/
  maxxflow_core/       settings, clock, money (Decimal), hashing (HMAC), masterdata, ports
  maxxflow_data/       DAL (search_path isolation), DDL, transforms, schema definition
  maxxflow_synth/      synthetic generator, latent ground truth, 3 CI gates
  maxxflow_features/   medallion lake IO (fsspec)
  maxxflow_events/     in-process retrain queue + micro-batch debounce (Event Grid/Service Bus seam)
  maxxflow_mlops/      MLflow registry, BYOC serving router, promotion gate, drift
  maxxflow_providers/  LLM/embeddings adapters (deterministic stub + Azure OpenAI seam)
  maxxflow_cli/        the `maxxflow` command (each subcommand == one AML job)
modules/           m1_quote (reference), m2_inventory, m3_delay, m4_bom, m5_scheduling (stub)
services/          configurator (FastAPI), dashboard (Streamlit), train_csv CLI
frontend/          React + Vite + TypeScript Configurator UI
docker/            one multi-stage image family (base→src→train/serve/ui) + score.py (BYOC entry)
db/                generated tenant-template DDL (schema-per-tenant)
dataset/           prepared gold CSVs for M1 (gold_quote_win.csv, gold_price_band.csv)
tests/             unit · contract · data_quality · smoke · parity
docs/              WORKFLOW_COMPONENT_REFERENCE.md · CODEBASE_WALKTHROUGH.md
```

## Prerequisites

- **Docker** + Docker Compose (for the Postgres / MinIO / MLflow / serving stack).
- **Python 3.10–3.12** (see [Troubleshooting](#troubleshooting) for the 3.13 caveat).
- **[`uv`](https://docs.astral.sh/uv/)** for dependency management.
- **macOS only:** `brew install libomp` (LightGBM's OpenMP runtime).

## Quick start

```bash
# 1. Install the pinned environment (this same env is baked into the ACR image)
uv sync --extra dev --extra serve --extra ui

# 2. Bring up the stack and provision a tenant schema
make bootstrap                       # postgres + minio + mlflow up, tenant schema created

# 3. Generate data, validate it, and run the M1 slice end-to-end
make seed     TENANT=demo MODULE=all
make validate TENANT=demo MODULE=all
make fe train serve score drift MODULE=m1_quote TENANT=demo

# 4. (Optional) bring up the human-facing surfaces
make ui                              # Configurator API :8000 · dashboard :8501 · React UI :5173

# 5. Run the test suite
make test
```

## The inner loop

Each `make` target is intentionally 1:1 with an Azure ML pipeline step. Flip `APP_ENV=dev-azure`
in Phase 2 and the **same commands** run against the Azure read replica, ADLS, the Azure ML MLflow
workspace, and managed endpoints — with no code change.

| Command | What it does | Azure equivalent |
|---|---|---|
| `make bootstrap` | Stack up + `maxxflow db-provision` (tenant schema) | infra + migration |
| `make seed` | Synthetic ground truth → tenant Postgres | ADF ingest |
| `make validate` | 3 CI gates: schema / realism / leakage+learnability | data-quality gate |
| `make fe` | bronze → silver → gold features on MinIO | AML feature step |
| `make train` | LightGBM + isotonic → register to MLflow | AML training job |
| `make serve` | `azmlinfsrv` model-server (same image as the endpoint) | AML online endpoint |
| `make score` | advisory scores → `<entity>.customElements` | AML batch/endpoint |
| `make drift` | Evidently / PSI report | AML model monitor |
| `make test` | unit + contract + data-quality + smoke + parity | CI |

The `maxxflow` CLI backs these targets directly (`maxxflow seed|validate|fe|train|score|drift|serve --tenant <t> --module <m>`).

## Smart Quote Optimiser (M1) — UI & training

M1 ships two per-tenant models trainable from the prepared gold CSVs in `dataset/`, from an uploaded
CSV, or from the tenant's read replica:

- **Classification** (`t_<tenant>__m_m1_quote_win`) — LightGBM + isotonic calibration on
  `gold_quote_win.csv` (label `won`), with light auto-HPO. Metrics: accuracy, AUC, PR-AUC,
  precision/recall/F1, Brier, ECE, confusion matrix.
- **Regression** (`t_<tenant>__m_m1_quote_price`) — three LightGBM quantile regressors (P25/P50/P75)
  on `gold_price_band.csv` (target `price_ratio`), **benchmarked against an empirical baseline**; the
  empirical band remains the *served* fallback unless the fitted models beat it on coverage **and**
  pinball loss. Recommended price is clamped to ±30% of list price.

```bash
# Train from the prepared CSVs (CLI)
make train-csv TENANT=furniched
python services/train_csv.py --publish            # trains the shared `global` base model + publishes
```

Bring up the UI (`make ui`) for the full Configurator workflow:

- **Configurator API** — FastAPI on `:8000` (`/docs`): model cards, `predict` (Test Predictions),
  the 5-step retrain wizard (`retrain/preview` validates columns; `retrain/db-columns` for the
  Connect-DB path), background `train` with **live logs**, and `publish` (champion gate — promotes
  only if the candidate beats the current version).
- **Dashboard** — Streamlit on `:8501`: models, pipeline status, per-model metrics
  (win: AUC/Brier/ECE; price: coverage/pinball/MAE + fitted-vs-empirical), storage, drift.
- **React UI** — Vite on `:5173`: the product-styled Configurator (tenant selector, cards, Test
  Predictions drawer, retrain wizard, publish modal).

## Multi-tenant model routing & the `global` base model

Models are registered under the name `t_<tenant>__m_<module>` — the **name itself is the routing
key**. Serving loads `models:/<name>@champion` only (no tag/alias search, which is unsupported on
Azure ML). Promotion is an alias move (`@champion`, with `@previous` kept for sub-minute rollback);
no redeploy.

A shared **`global`** tenant holds the base models we (developers) train on pooled data. A newly
onboarded tenant with no model of its own is automatically served and shown the `global` base
(`t_global__m_<module>`) until it trains and publishes its own champion, which then overrides the
base. Training is **tenant-agnostic**: the `tenant` value only names the registered model — it is
never a feature and never filters rows.

```bash
python services/train_csv.py --publish             # you: train + publish the shared base models
python services/train_csv.py --tenant furniched --publish   # a tenant: train its own from its data
```

## Configuration

Configuration is driven by `APP_ENV`, which selects a profile in `config/profiles/`:

| `APP_ENV` | File | Purpose |
|---|---|---|
| `local` (default) | `config/profiles/local.env` | Fully populated — local Postgres, MinIO, MLflow, stub providers |
| `dev-azure` | `config/profiles/dev-azure.env` | Phase-2 seam (Azure resources — `TODO` markers, verify before building) |
| `staging` / `prod` | `config/profiles/*.env` | Phase-2 stub seams |

`.env.local` (git-ignored) is layered last for machine-specific secrets. Module code reads typed
fields from `Settings` — it never inspects `APP_ENV` directly. Key fields include `DATA_DB_URL` /
`DATA_DB_IS_REPLICA`, `LAKE_URI`, `MLFLOW_TRACKING_URI`, `MODEL_ALIAS`, `LLM_PROVIDER` /
`EMBEDDING_PROVIDER` / `EMBEDDINGS_ENABLED`, `HMAC_SALT`, `PRESENTATION_TZ`, and `MODEL_LRU_SIZE`.

## Service ports

| Service | URL | Notes |
|---|---|---|
| PostgreSQL | `localhost:5432` | tenant schemas `tenant_<slug>` |
| MinIO | `:9000` / console `:9001` | buckets `maxxflow-lake`, `mlflow-artifacts` |
| MLflow | http://localhost:8085 | tracking + registry (`--serve-artifacts`) |
| Model server (azmlinfsrv) | http://localhost:5001 | BYOC router for M1/M3/M4 |
| Configurator API | http://localhost:8000/docs | FastAPI |
| Dashboard | http://localhost:8501 | Streamlit |
| React UI | http://localhost:5173 | Vite |

## Testing

```bash
make test                                  # full suite (~1–2 min on a laptop)
uv run --extra dev pytest tests/unit -q     # a single group
```

The suite is grouped and each test protects an invariant:

- **unit** — Decimal money math, TZ-invariant clock, HMAC tiers, per-module guardrails, M3 leakage.
- **contract** — generator output columns == DAL SELECT columns; feature-frame shape.
- **data_quality** — the 3 gates pass for every module, and *genuinely fail* on a planted leak / an AUC≈1.0 model.
- **smoke** — each module trains → registers `@champion` → serves → scores.
- **parity** — no `if env==` branches, registry name+alias only, azmlinfsrv serving, train↔serve round-trip.

DB-backed tests are marked `needs_db` and auto-skip when no Postgres is reachable, so the suite runs
green both before and after `make bootstrap`.

## Design principles (non-negotiables)

These are enforced in code and by tests, not just documented:

- **Tenant isolation via `search_path`** — never `WHERE tenant_id`. A read guard rejects any SQL
  containing `tenant_id` or the stale `rop_status`.
- **Classical ML only** — no transformer hosted/trained; LLM/embeddings are external APIs behind
  ports, with deterministic stubs (never a paid API in tests).
- **`decimal.Decimal`** for all margin / ±30% clamp / 110% overrun math — no float drift at boundaries.
- **One TZ-explicit as-of clock** — identical features regardless of the process timezone.
- **PII never leaves the DAL** — operator UUIDs are HMAC→skill-tier before bronze.
- **Derive, don't read** — ROP state is derived live; GRN on-time keys off the status-transition
  timestamp, not `createdAt`.
- **No leakage** — M3 features are censored as-of-T; a gate fails any model that looks *too good*
  (AUC≈1.0), because that means a label leaked.
- **Registry: name + `@champion` only** — no search/stage APIs (they fail silently on Azure ML).
- **BYOC serving** — `azmlinfsrv` locally and on Azure, through one scoring script.

`NOTES.md` §5 maps each rule to the exact file that enforces it.

## Local ⇄ Azure parity

Every local component has a production twin, chosen by config profile — Phase 2 is *additive*, not a
rewrite:

| Local | Azure (Phase 2) |
|---|---|
| PostgreSQL | Azure PostgreSQL **read replica** |
| MinIO / S3 | ADLS Gen2 (`abfss://`) |
| Self-hosted MLflow | Azure ML workspace MLflow |
| `azmlinfsrv` container | AML managed online endpoint |
| In-process event queue | Event Grid / Service Bus |
| Stub LLM / embeddings | Azure OpenAI / Foundry |
| `make` targets | Azure ML pipeline jobs |

## Documentation

| Document | Purpose |
|---|---|
| [`docs/WORKFLOW_COMPONENT_REFERENCE.md`](docs/WORKFLOW_COMPONENT_REFERENCE.md) | Every architecture component — what / why / how |
| [`docs/CODEBASE_WALKTHROUGH.md`](docs/CODEBASE_WALKTHROUGH.md) | Per-file, per-function reference for the whole repo |
| [`NOTES.md`](NOTES.md) | Assumptions, verification status, and the Phase-2 (Azure) go-live checklist |
| [`SDD_v1.2_change_list.md`](SDD_v1.2_change_list.md) | Change list vs the Software Design Document |
| [`db/README.md`](db/README.md) | How the tenant DDL is generated |
| [`frontend/README.md`](frontend/README.md) | React UI dev notes |

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `numpy` build fails during `uv sync` | You're on **Python 3.13** — numpy 2.0.x has no cp313 wheels. Use 3.10–3.12 (`.python-version` pins 3.12). |
| `OSError: libomp.dylib` (LightGBM, macOS) | `brew install libomp`. |
| MLflow UI unreachable on `:5000` | `:5000` is macOS AirPlay; this project uses **`:8085`**. Some ports (e.g. 5060) are blocked by Chrome. |
| Docker build fails pulling `ghcr.io/astral-sh/uv` (TLS timeout) | Fixed — the Dockerfile installs `uv` from PyPI. Just re-run `make ui` / `make serve`. |
| `.venv/bin/python: No such file` in a container/sandbox | `.venv` is host-specific; rebuild the env in that environment (don't copy a macOS `.venv` into Linux). |
| "only one class present — cannot train" on a CSV | The dataset's `tenant` column values don't match — training is tenant-agnostic now; pull the latest and re-run. |

## Roadmap

- **M5 — Job Scheduling (Phase B):** OR-Tools CP-SAT constraint scheduler + a deferred RL layer.
  Currently an explicit stub that raises `NotImplementedError` and is not wired into the CLI.
- **Phase 2 — Azure:** Bicep infra (AML workspace, ADLS, ACR, Postgres+replica, Key Vault, ADF,
  Event Grid, Service Bus, Monitor), thin AML SDK v2 pipeline wrappers that call the same Phase-1
  functions, and deployment of `score.py` to a managed online endpoint. See `NOTES.md` §4 for the
  credential checklist.

## Contributing

1. Create a feature branch and keep changes small and reviewable.
2. Run `make test` — it must stay green (parity guards will fail the build on `if env==`, registry
   search, or non-azmlinfsrv serving).
3. Keep the [non-negotiables](#design-principles-non-negotiables) intact; add a test when you touch a
   guardrail or a leakage-sensitive path.
4. The intended checkpoint commit sequence is scripted in `scripts/commit_checkpoints.sh`.

## License

Proprietary and confidential. © Avonet Technologies / MaXXFlow. All rights reserved. *(Update this
section to your organisation's chosen license before external distribution.)*
# MLStudio
