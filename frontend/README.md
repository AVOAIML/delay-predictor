# Configurator Frontend — M3 Production Delay demo

A small React + Vite + TypeScript page that shows what M3 (Production Delay)
puts in front of a planner on a Manufacturing Order: the total estimated
delay, a risk badge, the "why" explanation lines, and the material-overrun
list. It reproduces the target screenshot's MO-creation layout end to end
(breadcrumb, stage stepper, component availability table, activity log), and
can run **either** against bundled fixtures **or** against the real M3
pipeline over a live local Postgres.

## Run it (fixtures only, no backend)

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

Use the **Demo MO** dropdown in the top bar to switch between four
fixtures that each exercise a different M3 signal:

| Order | Signal fired | Risk |
|---|---|---|
| `WH/MO/00142` Wooden Table | `time_overrun_ratio` | High — this is the screenshot scenario, and its fixture is a byte-for-byte copy of a real pipeline run (see below) |
| `WH/MO/CASCADE-001` Critical-path Assembly | `critical_path_cascade_ratio` | Medium — a not-started dependent inherits a 1.30× critical-path overrun |
| `WH/MO/00151` Steel Cabinet | `operator_pace_ratio` + `material_shortfall_ratio` | Medium |
| `WH/MO/00163` Office Chair | none fired | Low, no why-lines |

## Run it against the real M3 backend

This repo's `db-provision` already creates a local Postgres tenant schema
with real MRP tables (`work_orders`, `mo_components`, `items`, …) that M3's
own rule engine reads — no synthetic generator exists for M3 (its module
docstring says so explicitly: it reads live MRP data, there's nothing to
synthesize). So "real backend" here means: seed one real job, run the real
scoring/review pipeline, then read back what it actually computed.

```bash
# 1. seed one job into tenant_demo (idempotent, safe to re-run)
uv run python scripts/m3_demo_seed.py

# 2. run the REAL pipeline: rule engine -> weight agent -> review agent -> publish
uv run python -m m3_production_delay.review --tenant demo --threshold 1.0 \
  --job "WH/MO/00142" --job "WH/MO/CASCADE-001"

# 3. serve that row over HTTP, unauthenticated, for local use only
uv run uvicorn scripts.m3_demo_api:app --port 8010

# 4. point the frontend at it
echo "VITE_API_BASE=http://localhost:8010" >> frontend/.env.local
echo "VITE_TENANT=demo" >> frontend/.env.local
cd frontend && npm run dev
```

With `VITE_API_BASE` set, the frontend loads its complete active-order list
from `GET /api/{tenant}/demo-manufacturing-orders`. Product, BOM, quantity,
scheduled date, status, components, Work Orders, activity timestamps, and the
cached M3 insight all come from tenant-scoped database rows. Bundled fixtures
are used only when no API base is configured; live mode does not mix fixture
orders with database orders.

### Creating a new MO from the UI

With the bridge running, a **"+ New MO"** button appears next to the Demo MO
dropdown (it's hidden in fixture-only mode — there's nothing for it to call).
It opens a database-driven form. Selecting a Product loads that Product's
BOMs; selecting a BOM loads its component defaults and operation/dependency
graph. Changing the MO quantity scales the BOM component requirements. You
can then adjust component requirements and stock,
add custom components, enter actual progress on the loaded Work Orders, add
new Work Orders, and choose predecessor relationships before submitting. The
form then calls
`POST /api/{tenant}/demo-manufacturing-orders`
(`scripts/m3_demo_api.py`), which:

1. inserts the configured MO, component snapshots, Work Orders, time logs,
   and custom operation dependencies into `tenant_demo`'s real MRP tables,
2. runs `m3_production_delay.review.pipeline.run()` against it for real, and
3. returns the freshly computed `ValidatedInsight`.

After scoring, the POST route reloads the newly created MO using the same
database reader as the initial page load and returns that persisted read
model. React does not reconstruct the product, BOM, components, Work Orders,
or activity locally. The returned MO is selected immediately. There's no MO
CRUD endpoint on the real Configurator
API (M3 scores existing MOs, it doesn't create them — MO creation belongs to
the main MaXXFlow product's own UI), so this only works against the local
bridge, never against a live authenticated deployment.

**Why a separate bridge (`scripts/m3_demo_api.py`) instead of the real
`services/configurator/app.py`:** that app's actual `GET
/api/{tenant}/delay-insights` route is real and reads the exact same
table/column — but it sits behind `services/configurator/security.py`'s JWT
+ DB-backed RBAC, which queries `public.master_data`, `public.user_tenants`,
`public.roles`, `public.permissions`, and `users.is_ci_admin`/`users.upn`.
None of those exist locally: this repo's own DDL
(`libs/maxxflow_data/schema_def.py` `PUBLIC_TABLES`) only ever creates
`public.users`/`public.tenants`. That auth schema belongs to the main
MaXXFlow product's shared database that this AI/ML repo sits beside in
production — it isn't something `db-provision` here creates, so there is no
way to obtain a bearer token this API would accept without hand-building
unrelated product infrastructure. `scripts/m3_demo_api.py` is a byte-for-byte
copy of the real route's SQL, minus that auth middleware, clearly commented
as local-only. It is not a replacement for the real endpoint and shouldn't
be pointed at anything but a local dev Postgres.

Both `scripts/m3_demo_seed.py` and `scripts/m3_demo_api.py` are local dev
tooling in the same spirit as the `scripts/dev_auth.py` /
`scripts/seed_m3_demo.py` this repo's `.gitignore` already excludes by name
(see its "Local-only developer tooling" block) — they're left untracked
rather than added to that block; add them there yourself if you'd rather
they never show up in `git status`.

## What's real vs. presentational

- `src/types.ts` mirrors `ValidatedInsight` / `InsightLine` / `ComponentEvidence`
  from
  [`modules/m3_production_delay/review/schemas.py`](../modules/m3_production_delay/review/schemas.py)
  field-for-field, confirmed against an actual pipeline run (not guessed) —
  down to `signal_key` values carrying the rule engine's `_ratio` suffix
  (`time_overrun_ratio`, not `time_overrun`).
- `risk_score` is an **unbounded** weighted mean of raw ratios, not a 0–1
  probability — a job 88% over its planned duration with three short
  components scores `2.23` against a `delay_threshold` of `1.0`. `src/riskBand.ts`
  bands it as "% of delay threshold" (`risk_score / delay_threshold * 100`,
  ≥150% = High, else Medium, or Low when `is_delayed` is false) — this
  framing, like the High/Medium/Low labels themselves, is this frontend's own
  presentational choice, not something the module returns.
- The default fixture's operation reads as "INDEPENDENT · 6e117fe4" rather
  than "Assemble Table" because `build_job_rollups()`
  (`modules/m3_production_delay/rule_engine/rollup.py`) reads
  `operations."operationName"` from Postgres but never carries it through
  into the per-operation dict the evidence/composer layer sees — confirmed
  by reading that file, not a display bug here.
- The headline copy is the deterministic composer's own templates in
  [`modules/m3_production_delay/review/composer.py`](../modules/m3_production_delay/review/composer.py)
  (`HEADLINES`, `_time_line`, `_operator_line`, `_material_line`) — what the
  Review Agent actually renders for a fired, weighted signal, not invented
  copy.
- `status: "fallback_template"` on the live run is expected locally: the
  Review Agent's judge call goes through `LLM_PROVIDER=stub`
  (`config/profiles/local.env`), which returns deterministic text, not JSON,
  so the judge rejects it and the deterministic template lines ship
  unjudged — see `modules/m3_production_delay/orchestrator.py`'s module
  docstring.
- Everything **around** the delay panel — the MO form, component table,
  stepper, activity feed — is presentation scaffolding for this demo, not
  something M3 or the Configurator API serves; there is no MO CRUD endpoint
  here.

## Build

```bash
npm run build   # tsc -b && vite build -> dist/
```

`services/configurator/app.py` serves `frontend/dist` as static files when
present (see the `_DIST` mount at the bottom of that file), so a production
build drops straight into the same origin as the real (authenticated) API —
useful once the product's shared auth schema is actually reachable.
