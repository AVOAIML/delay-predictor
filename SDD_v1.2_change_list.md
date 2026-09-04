# MaXXflow Phase 3 AI/ML SDD — v1.1 → v1.2 Change List

Paste-ready replacement text to reconcile the SDD with the approved plan
(`okey-now-i-am-structured-quill.md`) and the delivered build.

**Scope assumption:** the CSV/Excel "ML Studio" upload pathway, column-mapping UI,
and `mlmodel_*` metadata tables are **not** in the delivered Phase-3 backend. Below
they are marked **Future scope**; the doc is otherwise aligned to the as-built
design (schema-connector + synthetic-first, `customElements` writeback, MLflow
registry). If you'd rather remove the upload studio entirely, delete §3, §5.2, and
the upload rows in §4/§5.1 instead of "future scope" labelling.

Legend: **REPLACE** = swap the section's content; **ADD** = new content.

---

## §1.2.1 In-Scope — REPLACE

- Four platform-trained predictive modules delivered this phase, embedded in the MRP screens:
  Smart Quote Optimiser (M1), Predictive Inventory Alerts (M2), Production Delay Predictor incl. T&M overrun (M3), AI-Assisted BOM Cleanup (M4).
- Intelligent Job Scheduling (M5) — **Phase B scaffold only** (OR-Tools constraint-model placeholder + RL note; not built this release).
- **Classical ML only** (LightGBM / scikit-learn / numpy-scipy / statsforecast). No transformer models are hosted or trained. Any language/semantic needs use external LLM/embedding APIs behind swappable provider interfaces (deterministic stub locally), never for scoring.
- Data source: **live tenant PostgreSQL schema** read directly by the DAL, plus a **synthetic-data generator** for cold-start. (CSV/Excel upload — *future scope*.)
- Model lifecycle: feature engineering → training → isotonic calibration → MLflow registry (champion/challenger via `@champion` alias) → guardrailed scoring → drift monitoring, with per-module automated retraining triggers.
- Multi-tenant isolation via **schema-per-tenant** (`SET search_path`): one model artefact per (tenant, module); no cross-tenant data.
- Advisory scores written back into existing `customElements` JsonB columns; suppressed/low-confidence outputs logged to `audit_logs`.

## §1.2.3 Assumptions — REPLACE

- MaXXflow is deployed as multi-tenant SaaS on **Microsoft Azure (Australia East)** for AU/NZ data residency.
- Each tenant operates in an isolated PostgreSQL **schema** (schema-per-tenant); the ML pipeline reads/writes the tenant schema directly via `SET search_path` — there is **no `tenant_id` column and no `WHERE tenant_id`** on business tables.
- **Cold-start is solved by a synthetic-data generator**, not a fixed history gate: it seeds MasterData codes then writes schema-faithful, label-bearing rows embedding a known latent ground-truth per module, so models train from day one. Per-module guardrails degrade gracefully on thin real data.
- The platform team maintains the shared ML infrastructure: **MLflow** (tracking + registry), **Azure ML** (training jobs + one always-on inference endpoint), **ADLS Gen2**, and the local docker-compose parity stack (Postgres + MinIO + MLflow + model-server).
- No model weights, training data, or feature statistics are shared across tenants.

## §1.3 Methodology — REPLACE

Delivered as a parity-first, local-then-Azure sequence. Local build checkpoints:
(0) parity skeleton — config profiles, ports/adapters, `compose.local.yml`, one multi-stage Dockerfile from a single `uv.lock`, local MLflow, stub train≡serve proof;
(1) DAL + synthetic generator + 3 validation gates;
(2) Module 1 full vertical slice end-to-end;
(3) fan out M2/M3/M4 + the M5 scaffold.
Azure is Phase 2 (additive) — the parity seams are built now; only local adapters are implemented.

## §1.4 Acronyms — EDIT

- **Remove:** LSTM, NLP, RLS, CDR (not used in this design).
- **Keep:** API, BOM, CSV, ETL, MLOps, MRP, OOTB, SDD, SKU, T&M, UoM.
- **Add:** ADLS (Azure Data Lake Storage Gen2), ADF (Azure Data Factory), MLflow, azmlinfsrv (azureml-inference-server-http), EWMA (Exponentially-Weighted Moving Average), Croston (intermittent-demand forecast), TF-IDF (Term Frequency–Inverse Document Frequency), HMAC, ROP (Reorder Point), GRN (Goods Received Note), MO (Manufacturing Order), WO (Work Order).

---

## §2.2.2 Five model types — REPLACE the Algorithm and Min-data columns

| Model | Primary output | Algorithm (classical only) | Cold-start signal |
|---|---|---|---|
| Smart Quote Optimiser (M1) | Calibrated win prob 0–100% + recommended price band | LightGBM + **isotonic calibration**; price band = quantile band over comparable *won* quotes | Synthetic latent win function |
| Predictive Inventory Alerts (M2) | Stockout prob at 30d & 60d | Chronological weekly hazard candidates (**Logistic Regression / Random Forest / LightGBM / XGBoost**) → weekly calibration → constrained 30d/60d horizon calibration + deterministic rules. **Weekly batch.** | Weekly inventory snapshots |
| Production Delay Predictor (M3) | P(delay) at 25% milestone + T&M overrun hours | **Two LightGBM heads** (classifier + regressor); Bayesian target-encoding for work centre | Synthetic WOs + time logs with delay drivers |
| AI-Assisted BOM Cleanup (M4) | Per-line error prob + suggested fix | Rule engine + **robust z-score (median/MAD) + TF-IDF char n-grams + RapidFuzz** + LogisticRegression/LightGBM combiner (surface > 0.35) | Rules + unsupervised anomaly at 0 corrections |
| Intelligent Job Scheduling (M5) | *Phase B — scaffold only* | OR-Tools CP-SAT placeholder + RL note (not built) | — |

> Note: "Min. data" thresholds are replaced by the synthetic-first cold-start plus per-module guardrails (see §7.2). No transformer (Prophet/LSTM/BERT/PyTorch) is used anywhere.

## §2.3 Required Environment — REPLACE (full table)

| Component | Specification |
|---|---|
| Application stack | Next.js (MRP UI), NestJS (API layer, reads `customElements` scores), **PostgreSQL 16** (tenant schemas). *App layer is separate from the ML backend below.* |
| ML runtime | **Python 3.10–3.12** (pinned 3.12 locally): scikit-learn, LightGBM, numpy, scipy, RapidFuzz, pandas, pyarrow, fsspec/s3fs, SQLAlchemy, psycopg, MLflow, pandera. **No PyTorch/Prophet.** |
| ML logic | Plain Python functions + a CLI/Makefile so the same code runs locally and as Azure ML components unchanged. |
| Orchestration | **Azure ML SDK v2 pipelines**, triggered by **Azure Data Factory** (schedules) and **Event Grid / Service Bus** (events). Locally: Makefile/CLI targets + an in-proc queue/APScheduler stub. **(No Airflow.)** |
| Model registry / tracking | **MLflow** (Azure ML workspace is MLflow-compatible): experiment tracking, versioning, champion/challenger via `@champion` **alias**. Backend store = Postgres; artifacts = object store (MinIO local / ADLS prod) with `--serve-artifacts` proxy. Local UI port **8085**. |
| Inference | **One always-on Azure ML online endpoint** running a **BYOC multi-model router under `azureml-inference-server-http` (azmlinfsrv)** — one scoring script, routes by registered-model **NAME + `@champion`**, LRU-loads per (tenant, module). **M2 is a batch job, not on the endpoint.** Locally the `model-server` container runs the same image (port 5001). |
| Feature store / lake | **Medallion lake (bronze/silver/gold) on fsspec** — MinIO locally == **ADLS Gen2** in prod (only the URI protocol differs). Features engineered through the DAL. **(No Redis.)** |
| Secrets | **Azure Key Vault** (prod) / `.env.local` (local), selected by pydantic-settings profiles. |
| Containerisation | Docker; one multi-stage image family (base→train→serve) from a single `uv.lock`; pushed to **ACR**; deployed to the Azure ML endpoint. Local: docker-compose. |
| Cloud platform | **Microsoft Azure, Australia East** (AU/NZ residency). |
| Monitoring / drift | **Evidently** drift reports (PSI fallback) + **Azure Monitor**. |

## §2.4 Constraints — REPLACE (key edits)

- Data residency: all tenant data, model artefacts, and logs remain in **Azure Australia East**.
- Tenant isolation: **schema-per-tenant via `search_path`**; per-tenant model artefacts; no cross-tenant sharing.
- Cold start: solved by the **synthetic-data generator** + per-module guardrails (not a 12-month unlock gate).
- Classical-ML-only: no transformer models hosted/trained; external LLM/embeddings only via provider interfaces, never for scoring.
- Privacy: any feature derived from operator/employee PII is **pseudonymised (HMAC → skill tier) inside the DAL before it reaches bronze / MLflow / any external API**.
- *(File-upload / virus-scan / 200 MB constraints → move to a "Future scope" note.)*

---

## §4.1 Architecture Layers — REPLACE

| Layer | Components |
|---|---|
| Config & ports/adapters | `pydantic-settings` profiles (local / dev-azure / staging / prod). Cloud behaviour sits behind ports/adapters chosen by config — no `if env==…` in module code. |
| DAL (`maxxflow_data`) | The one place that reads/writes tenant Postgres. Sets `search_path`; forbids `tenant_id`/`rop_status`; single as-of clock; Decimal math; operator HMAC→tier before bronze. |
| Feature store / lake (`maxxflow_features`) | Medallion read/write on fsspec (MinIO==ADLS). |
| Providers (`maxxflow_providers`) | `LLMProvider` / `EmbeddingProvider` — deterministic stub locally; Azure OpenAI/Foundry in Phase 2. |
| MLOps (`maxxflow_mlops`) | MLflow registry (name + `@champion` only), champion/challenger promotion gate, BYOC serving router, drift. |
| Synthetic engine (`maxxflow_synth`) | Domain simulators + loader + the 3 validation gates. |
| Modules (`m1_quote` … `m5_scheduling`) | Per-module `features / train / score / dal / pipeline`. |
| Serving | azmlinfsrv BYOC multi-model router (M1/M3/M4 online; M2 batch). |

## §4.2 End-to-End Data Flow — REPLACE

1. **Seed** — synthetic generator seeds MasterData + schema-faithful, label-bearing rows into the tenant schema (== ADF ingest in prod).
2. **Feature engineering** — DAL reads the tenant schema → shared transforms → gold parquet on the lake.
3. **Train** — each module's canonical calibrated candidate pipeline → logged & registered to MLflow.
4. **Promote** — champion/challenger gate moves the `@champion` alias if the challenger beats it on holdout + passes calibration.
5. **Serve** — M1/M3/M4 via the azmlinfsrv endpoint (route by name + `@champion`); M2 via weekly batch.
6. **Writeback** — advisory scores → `customElements` (M1→Quotation, M2→Item, M3→ManufacturingOrder, M4→Bom); low-confidence/suppressed/dismissed → `audit_logs`.
7. **App** — NestJS reads the cached `customElements` value; the MRP widget renders it (no blocking AI call).
8. **Monitor** — Evidently drift report; delayed-label performance tracked as outcomes arrive.

---

## §5.1 Data Model — REPLACE

- **Tenant isolation is schema-per-tenant; there is no `tenant_id` column and no RLS.** Remove the "each table carries a `tenant_id` column backed by RLS" statement.
- The delivered pipeline does **not** create `mlmodel_*` tables. It uses:
  - existing `customElements` JsonB columns for advisory writeback (Quotation / Item / ManufacturingOrder / Bom);
  - the existing `audit_logs` table for suppressed/low-confidence/dismissed events;
  - **MLflow** (not a DB table) as the model registry; run/version metadata lives there.
- **Synthetic labels** persisted where columns can't derive them: M1 from stage/status UUIDs, M3 from `realDuration`; **M2/M4 labels stored under `customElements.__synthetic_label__`**.
- *(The `mlmodel_uploads/schema_conns/mappings/…` tables belong to the future CSV-upload studio — mark as future scope.)*

## §5.2 File Upload Pipeline — MARK "Future scope" (not built this phase)

## §5.3 Schema Connector Feature Templates — KEEP, minor edits

- Correct M3 template to as-built as-of-T features: pace-so-far at the 25% **physical** milestone (leakage-censored time logs), work-centre concurrency ÷ `len(allowedEmployees)` (0-guarded), complexity (BOM components + operations + dependency depth), operator **skill tier (HMAC)**, material shortfall. Overrun label from `realDuration` vs 110% of expected.
- Correct M2 template: consumption from `MOComponent.consumedQty` increments; vendor on-time keyed off the **GRN status transition** (not `createdAt`); derive ROP live.

---

## §6.1 Training & Versioning — REPLACE

- Every run is versioned in MLflow under `t_<tenant>__m_<module>`.
- **Champion/challenger promotion is automated via the `@champion` alias**: promote only if the challenger beats the champion on holdout **and** passes calibration (Brier not worse, ECE ≤ τ) **and** has no data-validation failures; otherwise keep champion and alert. Rollback = alias move to `@previous` (sub-minute, no redeploy). (Manual promote can remain an optional override.)
- One `@champion` per (tenant, module). Routing is by **name + alias only** — no tag/alias search, no stage-based deploy.

## §6.2 Retraining Triggers — REPLACE (three modes)

| Mode | Description | Applies to |
|---|---|---|
| Manual | "Retrain now" from the UI. | All |
| Scheduled | ADF schedule → Azure ML pipeline. | **M1 monthly, M2 weekly** |
| Event-driven | Trigger on a business event, debounced/micro-batched, with the label carried in the event envelope (replica-lag safe). | **M3** on MO→Done (Event Grid); **M4** on confirmed corrections / nightly (Service Bus) |

## §6.3 Performance Metrics — EDIT metric names

- M1: AUC-ROC, **Brier + ECE** (calibration), recommended-price band coverage.
- M2: PR-AUC / precision@threshold, reliability (calibration) band.
- M3: PR-AUC + F1, **overrun MAE** (regressor head).
- M4: **precision@k**, user-acceptance rate, false-positive rate.

---

## §7.1 Data Security & Tenant Isolation — REPLACE

- Isolation is **schema-per-tenant via `SET search_path`**; cross-tenant joins are physically impossible. **Remove the RLS/`tenant_id` statements.**
- Secrets in **Azure Key Vault**; endpoint auth via the Azure ML endpoint + tenant-scoped model **name** (defense-in-depth `tenant` tag asserted after load, never queried).
- Object storage is **ADLS Gen2** with per-tenant paths (MinIO locally).

## §7.2 Model Output Guardrails — REPLACE with as-built per-module rules

- **M1:** hide if <5 comparable historical quotes; "low confidence" flag if score <40% or >95%; price band clamped to ±30% of base `Product.salesPrice` (Decimal).
- **M2:** <6 months GRN history → rule-based estimate (not ML); **suppress** the alert if an open PO covers the deficit AND vendor on-time >95% (suppressed → `audit_logs`).
- **M3:** do not score below 20% completion; tenant-configurable delay threshold (default 60%); hide gracefully if scoring is unavailable.
- **M4:** surface only if error prob > 0.35; **never auto-corrects** (every change needs explicit Confirm).

## §7.3 Audit Logging — EDIT

- Training runs tracked in **MLflow**; suppressed/low-confidence/dismissed predictions logged to the tenant `audit_logs` table (module, action, entity, metadata JSON, timestamp).

## §7.4 Access Control — KEEP (fine as written)

---

## ADD — new subsections under §2 or §7

### Synthetic-Data Subsystem (cold-start engine)
Hand-written, seedable domain simulators (not SDV/CTGAN) that (a) respect every FK/constraint in `schema.prisma`, (b) embed a known latent ground-truth per module, (c) inject realistic noise, class imbalance and seasonality. The generator seeds `MasterDataCategory` + `MasterData` first (labels are defined by which MasterData UUID is written), then writes label-bearing rows into the tenant schema — the same path production reads. Graduation path: synthetic-only → hybrid → real-only per tenant, with MLflow provenance tags.

### The 3 CI Validation Gates
Before any synthetic batch trains: (A) **schema/constraint** (types, Decimal scale, FK existence, MasterData codes, `ropStatus` forbidden, `deletedAt` exclusion); (B) **statistical realism** (class-balance bands, distributions, seasonality); (C) **leakage + learnability** — no single feature's AUC exceeds a threshold, **trivial-model AUC ≈ 1.0 is a leak alarm (fails)**, and each module lands in an acceptance band (M1 AUC 0.72–0.82; M2 0.74–0.90; M3 0.68–0.90 + positive rate 0.15–0.40; M4 0.78–0.985 + precision@k).

### Schema-Fidelity & Leakage Integrity Controls
- Tenant isolation via `search_path` (never `WHERE tenant_id`).
- Derive ROP live; never read the stale `Item.ropStatus`.
- `decimal.Decimal` for all margin / ±30% clamp / 110%-overrun math.
- Single TZ-naive as-of clock (store UTC, present Australia/Sydney).
- GRN vendor on-time keyed off the status transition (`updatedAt`/`billCreatedAt`), not `createdAt`; null scheduled date = unknown (excluded).
- `len(allowedEmployees)` with a 0-guard (never divide by the array).
- **M3 as-of-T leakage censoring** with an `assert_no_leakage` check (forbidden-at-T columns: `realDuration`, `actualStart/End`, final `unitsDone`, terminal status, `MO.completedAt`).
- Operator UUIDs → HMAC → skill tier inside the DAL before bronze (no PII downstream).

---

## Phase-2 (Azure) items to mark "planned, not built"
`infra/` Bicep; Azure ML pipeline submission; ADF schedule + Event Grid/Service Bus wiring; swap stub providers for Azure OpenAI/Foundry. **Verify-before-building:** the LLM model id (Foundry Claude Sonnet 4.6 or Azure OpenAI deployment) and `text-embedding-3-small` AU-East deployability — mark as TODO, do not assume.
