# MaXXFlow EPIC 10 — Build Notes (assumptions + go-live checklist)

This file lists (1) every assumption baked into the build, (2) every spot that
needs **your Azure credentials / decisions** to go live, and (3) what was verified
here vs. what must run on your machine. Phase 2 (Azure) is intentionally **not**
authored (no `infra/` Bicep, no AML pipeline submission, no ADF/Event-Grid/Service-Bus
wiring) per the brief — the parity *seams* exist so those drop in additively.

---

## 1. Verification status (important — read first)

This session's sandbox had **no Docker, no Postgres, and no write access to git**.
So I could not run `make bootstrap` / `docker compose` / write to Postgres here.
What I **did** verify (66 tests + all 4 validation gates green, run with the pinned
env at `uv.lock`):

- Parity skeleton: config profiles, ports/adapters, the trivial-stub
  train→register→**serve-through-`score.py`** round-trip, and all `tests/parity`
  guards (no `if env==`, registry name-only, azmlinfsrv serving, single scoring script).
- DAL transforms (ROP derive, GRN on-time off status transition, `len(allowedEmployees)`
  0-guard, operator HMAC, M3 as-of-T censoring), Decimal math, TZ-invariant clock.
- Synthetic generator for **all four** modules + the **3 CI gates** (schema, realism,
  leakage+learnability with per-module AUC bands; AUC≈1.0 fails — proven by a negative test).
- Full per-module vertical slices in-memory: features → train (LightGBM / isotonic /
  two heads / combiner) → register by name + `@champion` → serve via the BYOC
  `ModelRouter` → guardrails → drift. Plus M3 `assert_no_leakage` and the M3/M4
  event-worker debounce.

What needs **your Docker stack** to exercise (code is written + structured for it,
but unrun here): `make bootstrap` (compose up + `db-provision`), `make seed` (the
Postgres **write**), `make fe` (DAL **read** → lake), `make score` (writeback to
`customElements`), `make serve` (the `model-server` container under azmlinfsrv),
`make drift`. The generator writes the **same columns** the DAL SELECTs (enforced by
`tests/contract/test_schema_contract.py`), so the DB path is wired to match.

To run the DB-backed tests after `make bootstrap`, they auto-enable (they're marked
`needs_db` and skip only when no Postgres is reachable).

> Tip: `make test` runs the whole suite; on a laptop it takes ~1–2 min (M3's
> two-head train is the slow part). All green in CI.

---

## 2. Git / commits

The sandbox could not write to `.git` (the host holds `.git/index.lock`), so I could
not create the checkpoint commits myself. The intended **small, reviewable commit
sequence** is scripted in `scripts/commit_checkpoints.sh` — run it from a clean
checkout to materialise the four checkpoints (0 parity skeleton → 1 DAL+synth+gates →
2 M1 slice → 3 M2/M3/M4+M5), or just `git add -A && git commit`.

---

## 3. Assumptions made

1. **Tenant-template DDL is the AI/ML-relevant subset.** `db/`/`schema_def.py`
   materialise the tables M1–M4 read/write (+ minimal `public.users`/`tenants`).
   Integration-only columns (`integration_*`, `stripe_*`, `superset_*`) and tables
   unrelated to the modules are omitted from the LOCAL DDL. **In prod the full schema
   is provisioned by your Prisma migrate**, not by this DDL.
2. **MasterData codes/categories** (`maxxflow_core/masterdata.py`) are a canonical set
   chosen to match the schema's FK categories (QUOTATION_STAGE/STATUS, MO_STATUS,
   WORK_ORDER_STATUS, PRODUCT_TYPE, etc.). IDs are deterministic `uuid5` for
   reproducibility. **Confirm these codes match your real tenant MasterData seeding**
   (esp. the win/loss codes: Win = stage `SALES_ORDER` / status `CONFIRMED` + `salesOrderId`;
   Loss = status `CLOSED_LOST`).
3. **M3 writeback target.** `work_orders` has no `customElements` JsonB, so M3 writes
   its advisory to the owning **`manufacturing_orders.customElements`** under
   `ai_delay[work_order_id]`. If you'd prefer a column on `work_orders`, add a migration.
4. **M4 writeback target.** `bom_components` has no `customElements`, so per-line
   advisories are written to **`boms.customElements`** under `ai_bom[bom_component_id]`.
5. **Labels.** M1's label is read from the status/stage UUIDs (real). M2/M3/M4 labels
   are *future/quality* events, so the **synthetic generator draws them from the latent
   ground-truth** applied to the observed features. On real data the labels become:
   M2 = realised stockout at horizon; M3 = realised delay at MO completion (carried in
   the MO→Done event envelope); M4 = confirmed/dismissed correction (carried in the
   Service-Bus envelope).
6. **Operator skill tier** uses `HMAC(salt)` pseudonymisation with a **pinned salt**.
   The local salt is a placeholder — the **prod salt must equal it** (pull from Key
   Vault) or tiers won't be reproducible local↔prod.
7. **Forecasting libs.** EWMA + a simple Croston are implemented in numpy (CPU/RAM
   light per the brief). `statsforecast` is left as an optional extra/seam, not required.
8. **Drift.** Evidently is optional; a dependency-free **PSI fallback** runs when it's
   not installed, writing the same HTML/JSON report.
9. **M3 target encoding** for high-cardinality `work_center_id` is **fit at train and
   stored in the model** (applied at serve), never recomputed from unlabeled serve data.
10. **Learnability bands** (`maxxflow_synth/gates.py::BANDS`) are tuned for the
    synthetic generator (M1 AUC∈[0.72,0.82]; M2∈[0.74,0.90]; M3∈[0.68,0.90] + pos∈[0.15,0.40];
    M4∈[0.78,0.985] + precision@k). Re-tune when real data arrives (graduation path §5).
11. **Events** (M3/M4) use an in-process queue + a debounced micro-batch worker as the
    local stand-in for Event Grid / Service Bus. The authoritative payload + label are
    carried in the envelope (replica-lag discipline).

---

## 4. Needs your Azure credentials / decisions to go live (Phase 2)

All live in `config/profiles/dev-azure.env` as `TODO(...)` markers. **Do not guess
these — verify before building** (the brief's "verify before building" rule):

1. **Foundry / Azure OpenAI model id (`LLM_MODEL_ID`)** — the BOM suggested-fix /
   conversational LLM. The plan names *Azure AI Foundry Claude Sonnet 4.6 (Global
   Standard)* or an Azure OpenAI deployment. **VERIFY the exact deployed model/deployment
   name + that it's reachable from AU East** before wiring. `providers/llm.py`'s
   `AzureOpenAILLMProvider` raises `NotImplementedError` until then.
2. **Embeddings (`EMBEDDING_MODEL_ID`, `EMBEDDINGS_ENABLED`)** — optional M4 semantic
   boost via `text-embedding-3-small`. **VERIFY it is deployable in AU East before
   enabling** (plan §12a #12). Default path is classical TF-IDF (OFF), so this is N/A
   unless you turn it on.
3. **PostgreSQL read replica (`DATA_DB_URL`, `DATA_DB_IS_REPLICA=true`)** — replica FQDN
   + Entra/managed-identity auth. ML must read the **replica**, never the primary.
4. **ADLS Gen2 (`LAKE_URI` `abfss://…`)** — storage account + container + auth.
5. **Azure ML workspace MLflow URI (`MLFLOW_TRACKING_URI` `azureml://…`)** — subscription/
   RG/workspace. The registry routing is already name+`@champion`-only (AML-safe).
6. **Key Vault** — `HMAC_SALT` secret (must equal the local salt) + DB/API credentials.
7. **Not authored this phase (by instruction):** `infra/` Bicep (AML workspace, ADLS,
   ACR, Postgres+replica, Key Vault, ADF, Event Grid, Service Bus, Monitor); thin AML
   SDK v2 pipeline wrappers that call the **same** Phase-1 functions as components; ADF
   schedule/event triggers replacing the local stubs; pushing the image to ACR and
   deploying `score.py` to a managed online endpoint. The seams make all of this additive.

---

## 5. Quick map of where each must-honor rule lives

- search_path isolation / no `tenant_id`, `rop_status` forbidden → `maxxflow_data/engine.py`
- Decimal margins / ±30% / 110% → `maxxflow_core/money.py`
- single as-of clock (TZ-invariant) → `maxxflow_core/clock.py`
- operator HMAC→tier → `maxxflow_core/hashing.py` + `maxxflow_data/transforms.py` + each module's `dal.py`
- GRN on-time off status transition → `maxxflow_data/transforms.py` (`grn_on_time`/`vendor_reliability`)
- `len(allowedEmployees)` 0-guard → `transforms.work_center_capacity`
- M3 as-of-T leakage + `assert_no_leakage` → `transforms.censor_timelogs_as_of`, `tests/unit/test_m3_leakage.py`
- registry name+`@champion` only (no search) → `maxxflow_mlops/registry.py` + `tests/parity/test_registry_name_only.py`
- BYOC azmlinfsrv serving → `docker/score.py`, `docker/Dockerfile` (serve stage), `tests/parity/test_serving_uses_azmlinfsrv.py`
- replica-lag: label carried in event envelope → `maxxflow_events/queue.py`, M3/M4 `pipeline.py`
- 3 synthetic gates (AUC≈1.0 fails) → `maxxflow_synth/gates.py`, `tests/data_quality/`
