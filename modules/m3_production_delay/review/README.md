# M3 Section 3 — Review Agent & Validated AI Insight

The layer between the Risk Engine's numbers and the MO **AI Insights** panel.
It explains a scored job in the user story's words, checks that explanation
against the evidence deterministically, has one LLM call judge whether each
line is supported, and writes the result back to the manufacturing order.

**The LLM never produces a number, a signal, or a line of text.** It answers
yes/no about lines the deterministic composer already wrote. That is a
property of the code's shape, not of the prompt: `composer.py` is the only
writer of a line, and `JudgeVerdict` has no field that could carry one.

## Where the code lives

Section 3 is split the same way the Weight Agent is — the agent under
`llm_agents/`, everything deterministic beside it, and both composed by the
orchestrator:

| | |
|---|---|
| **`llm_agents/review_agent/`** | The agent. Config, verdict models, exceptions, tracer, and `ReviewAgent.judge()` — the one LLM call. |
| **`review/`** (this package) | Everything deterministic: evidence, composition, validators, the pipeline and the writeback. Calls no LLM at all. |
| **`orchestrator.py`** | Builds both agents and exposes `review_jobs()` beside `resolve_weights()` / `resolve_risk_weights()`. |
| **`prompt/review_judge_{system,user}.txt`** | The judge's prompts, beside the Weight Agent's two. |

The dependency runs `review/` → `llm_agents/review_agent/` and never back:
the agent is handed an evidence pack and a draft, reads them, and returns its
own verdict. That direction is what makes "no number is ever produced by a
language model" structural rather than a prompt instruction.

---

## Pipeline

```
  Section 1                 Section 2
  calculate_delay_          resolve_risk_weights()        threshold
  elements_for_jobs()       -> {signal: weight}           (explicit, no default)
          |                          |                        |
          +-------------+------------+------------------------+
                        v
               evidence.build_evidence()      derives what Section 1 does not expose:
                        |                     supplier_reliability, overrun_basis,
                        |                     contribution shares, the 25% scoring gate
                        v
                    EvidencePack ------------------------------+
                        |                                      |
               composer.compose()                              |
                        |                                      |
                   InsightDraft  (template lines, ordered by contribution)
                        |                                      |
            validators.validate(pack, draft) <-----------------+
                        |
            any error? --yes--> status "rejected", no lines  (a bug here, not retryable)
                        |
                        no
                        v
            ReviewAgent.judge()  ── ONE LLM call ──> JudgeVerdict
                        |
            approved? --yes--> "approved" / "approved_with_warnings"
                        |
                        no ──> drop the unsupported lines, judge once more
                                        |
                        still no ──> "fallback_template"  (full template set, unjudged)
                        v
                 ValidatedInsight
                        |
              publish.publish_insights()
                        v
   manufacturing_orders.customElements -> { "ai_delay_insight": { ... } }
   + audit_logs row for fallback_template / rejected / suppressed_not_scorable
```

---

## Input contract

`review_job(job, weights, threshold, agent)`

| Argument | What it is |
|---|---|
| `job` | One element of `calculate_delay_elements_for_jobs()`'s output — a `{job_id, operations[...]}` dict with the element keys attached. A raw `build_job_rollups()` job is rejected with an explicit error. |
| `weights` | The tenant's resolved `risk_weights` — the same vector that produced the scores. Keys are the rule engine's own signal names. It does **not** sum to 1.0 (the Weight Agent's `seasonality` share has no rule-engine equivalent), and nothing here assumes it does. |
| `threshold` | The composite score above which a job counts as delayed. **Required at every level** — there is no default anywhere in Section 3, because the same score means different things for different tenants. It must be the cutoff the engine scored with: the engine's `is_delayed` flags are baked into the job, so reviewing at a different one is refused with an explicit error rather than publishing a badge attributed to a number that never produced it. |
| `agent` | A `ReviewAgent`. The orchestrator builds one; tests inject a scripted `LLMProvider` into it. Its config owns the attempt budget, so the number of calls the pipeline can spend is stated in one place. |

### What Section 3 derives on its own side

Section 1 and Section 2 are frozen. Everything below is computed here, from
what they already publish:

| Derived | How |
|---|---|
| `supplier_reliability` | Mean of `components[].vendor.vendor_lead_time_ratio` over components that have one. The engine computes this privately and feeds it into the score without attaching it. |
| `overrun_basis` | `"quantity"` when there is progress to extrapolate from, else `"operator_pace"`, else `"none"` — so a summary never implies a quantity projection for a number that came from an operator's history. |
| `contribution` | `weight × value / (score × Σweight_active)`. The `Σweight_active` factor undoes the renormalisation `composite_risk_score` performs over non-None signals, so shares over one operation sum to **1**. |
| `is_scorable` | The user story's 25% gate, shared by all operations in an MO: `Σ(expected duration × completed quantity ratio) / Σ(expected duration) >= 0.25`. Ratios are clamped to 0–1; missing progress contributes zero. |
| Job summary | The engine has no roll-up. `risk_score` / `overrun_hours` are the **max over scorable operations** (not a sum — parallel work would double-count), `is_delayed` is `any`. Recorded as `summary_basis: "worst_operation"`. |
| Job-scoped material & supplier | `rollup.py` attaches the MO's whole component list to every operation, so those two signals repeat identically. They are de-duplicated by `component_id` and stated **once per job**; time and operator stay per operation. |

### Fire baselines

One table, `evidence.FIRE_BASELINES`, read by the composer, the validators and
the judge prompt. Strictly greater-than in every case:

| Signal | Fires above | Weighted? |
|---|---|---|
| `time_overrun_ratio` | 1.0 | yes |
| `operator_pace_ratio` | 1.2 | yes |
| `material_shortfall_ratio` | 0.0 | yes |
| `supplier_reliability` | 1.0 | yes |
| `predecessor_time_overrun_ratio` | 1.2 | **no** — context only, weight pinned to 0 |

---

## Output contract

`ValidatedInsight`, written under `manufacturing_orders.customElements` →
`ai_delay_insight`. Section 1's per-operation `ai_delay[<work_order_id>]` is
never touched: the `COALESCE(...) || jsonb` merge only replaces this one key.

```json
{
  "ai_delay_insight": {
    "job_id": "WH/MO/00142",
    "status": "approved_with_warnings",
    "risk_score": 1.4523809523809523,
    "overrun_hours": 8.666666666666666,
    "is_delayed": true,
    "summary_basis": "worst_operation",
    "delay_threshold": 1.0,
    "why_lines": [
      {
        "index": 0,
        "signal_key": "time_overrun_ratio",
        "scope": "operation",
        "operation_id": "wo-cutting-0001",
        "headline": "Actual time logged has exceeded expected duration",
        "detail": "INDEPENDENT · wo-cutti · 10.0 hrs logged of 8.0 planned",
        "delta": "+2.0 hrs",
        "contribution": 0.38,
        "quoted": {"actual_hrs": 10.0, "expected_hrs": 8.0, "delta_hrs": 2.0}
      }
    ],
    "material_overrun": [
      {"component_id": "item-steel-plate", "name": "Steel Plate 12mm",
       "required_quantity": 100.0, "available_quantity": 40.0,
       "shortfall_quantity": 60.0, "vendor_name": "Lanka Steel",
       "vendor_lead_time_ratio": 1.5}
    ],
    "issues": [{"check": "not_started_but_delayed", "severity": "warning", "message": "...", "ref": "..."}],
    "judge": {"approved": true, "lines": [...], "unsupported_claims": [], "omitted_signals": [],
              "parse_error": null, "skipped": false},
    "attempts": 1,
    "model_version": "m3-review-v1",
    "generated_at": "2026-09-21T12:00:00+10:00",
    "scored_at": "2026-09-21T12:00:00+10:00"
  }
}
```

`quoted` is the contract that makes the text checkable: every numeral rendered
into a line maps back to the evidence field it came from.

### Statuses

| Status | Meaning | Audited |
|---|---|---|
| `approved` | Every line judged supported, no warnings. | no |
| `approved_with_warnings` | Approved, but something is worth knowing — a plausibility warning, a dropped line, or a delayed job with nothing fired. | no |
| `fallback_template` | The judge could not be reached, read, or convinced. The **full** deterministic template set is published, marked unjudged. | yes |
| `rejected` | A deterministic validator failed — a bug in this package. No lines are published. | yes |
| `suppressed_not_scorable` | No operation cleared the 25% gate. Summary numbers only, no explanation. | yes |

`material_overrun` is published for every status, including `suppressed_not_scorable`:
a short component is a fact about the warehouse, independent of the badge.

### Two behaviours worth knowing

**`LLM_PROVIDER=stub` always produces `fallback_template`.** The stub provider
returns prose, not JSON, which parses as `approved=False` with a
`parse_error`. Setting `M3_REVIEW_AGENT_LLM_ENABLED=false` does the same thing
without spending a call. Both are the intended degraded path, and it is why
the offline test suite never sees an `approved` status without a scripted
judge.

**A delayed job can have no why-lines.** The composite is a weighted mean of
raw ratios, so one signal sitting between 1.0 and its own baseline (an
operator pace of 1.15) can carry the mean past the threshold alone. The badge
stands — it is the engine's — and the insight ships with a `no_fired_signal`
warning naming the strongest signal instead of inventing a cause.

---

## Validators

All deterministic, all run before the judge. Only `error` rejects.

| Check | Severity | What it catches |
|---|---|---|
| `signal_missing` / `signal_not_fired` | error | a line citing a signal with no evidence, or one below its baseline |
| `omitted_signal` | error | a fired, weighted, renderable cause with no line |
| `omitted_signal_unrenderable` | warning | fired and weighted, but the supporting fields are missing, so no honest sentence exists |
| `quoted_number_unknown` / `quoted_number_mismatch` | error | a line quoting a field the signal has not got, or a value the evidence denies (±2%, with a 0.05 absolute floor so correct rounding is not a misquote) |
| `unquoted_number` | error | a numeral in the prose that matches nothing the line quotes. Entity names (`WH/MO/00142`, `Steel Plate 12mm`) are stripped first, so a part number is never read as a figure |
| `summary_mismatch` / `threshold_inconsistent` | error | a badge that disagrees with the evidence or with the tenant's threshold |
| `zero_weight_signal_shown` | error | a signal that did not move the score presented as a cause — this is what keeps a cascading predecessor out of the why-lines |
| `forbidden_cause` | error | `breakdown\|quality hold\|rework\|approval\|inspection\|weather\|strike` — none is observable from any signal here |
| `implausible_duration` | warning | logged time over 20× planned: possible minutes/seconds mismatch |
| `not_started_but_delayed` | warning | a `NOT_STARTED` / `TO_DO` operation the engine scored as delayed |
| `no_fired_signal` | warning | delayed with nothing fired; names the strongest signal and its value |
| `cascading_predecessor` | info | the operation inherits a predecessor running over 1.2× |
| `non_finite` | warning | `math.inf` (a zero-stock component) coerced to `None` |

---

## Wiring the LLM

Normal use goes through the orchestrator, which builds the agent and resolves
the weights the scores were produced with:

```python
from m3_production_delay.orchestrator import ProductionDelayOrchestrator

orchestrator = ProductionDelayOrchestrator()
insights = orchestrator.review_jobs(scored_jobs, weights=weights, threshold=1.0)
```

The agent can also be built directly, and a scripted `LLMProvider` injected —
which is how every test drives it, and the same pattern
`orchestrator._DemoScriptedLLMProvider` uses for the Weight Agent:

```python
from m3_production_delay.llm_agents.review_agent import ReviewAgent
from m3_production_delay.review import review_job

insight = review_job(scored_job, weights, threshold, ReviewAgent())
```

`ReviewAgent` resolves its provider from `maxxflow_providers.get_llm_provider()`
— the same port the Weight Agent's two calls use — at temperature 0, seed 0,
`max_tokens=900`, logging provider, latency and sizes but never prompt or
response bodies. One `llm_provider` passed to the orchestrator covers every
LLM call M3 makes.

### Selecting a model

| Setting | Value |
|---|---|
| `LLM_PROVIDER` | `azure_ai` for Azure AI Foundry, `azure_openai` / `openai` for the OpenAI data plane, `stub` (default) for offline |
| `AZURE_AI_API_KEY` | Foundry key |
| `AZURE_AI_API_BASE` | `https://<resource>.services.ai.azure.com` (the `/models` inference route is appended when the base carries no path of its own) |
| `LLM_MODEL_ID` | the Foundry **deployment name**, exactly as the deployment list shows it — not the vendor's model name, and never guessed |

Foundry and Azure OpenAI are different data planes with different
credentials: the `AZURE_OPENAI_*` settings cannot serve a Foundry deployment,
which is why `LLM_PROVIDER=azure_ai` exists as its own adapter. Put the
credentials in `.env.local` (git-ignored), not in a profile.

Two switches control the agent itself: `M3_REVIEW_AGENT_LLM_ENABLED=false`
skips the call entirely (publishing `fallback_template`), and
`M3_REVIEW_AGENT_TRACE_ENABLED` / `M3_REVIEW_AGENT_TRACE_INCLUDE_CONTENT` are
the only path through which the judge's raw prompt or response is ever logged.

Prompts live in `modules/m3_production_delay/prompt/` beside the module's other
two: `review_judge_system.txt` (the five checks) and `review_judge_user.txt`
(evidence pack + indexed candidate lines).

---

## Running it

### One job, end to end (dry run, no writes)

```bash
uv run python -m m3_production_delay.review --tenant demo --threshold 1.0 --job "WH/MO/00142" --dry-run
```

Drop `--dry-run` to write to `manufacturing_orders.customElements`; drop
`--job` to review every job for the tenant. `--threshold` is always required.
The batch entry reads through Section 1's DAL, resolves weights through
Section 2's orchestrator, reviews, then publishes — mirroring
`m2_inventory.batch_scoring`.

Without a live Postgres, review a job in process instead:

```bash
uv run python -c "
import json, pathlib
from m3_production_delay.review import review_job
from m3_production_delay.llm_agents.review_agent import ReviewAgent
job = json.loads(pathlib.Path('modules/m3_production_delay/review/fixtures/section1_job.json').read_text())
weights = {'time_overrun_ratio': 0.40, 'operator_pace_ratio': 0.30, 'material_shortfall_ratio': 0.15, 'supplier_reliability': 0.05}
insight = review_job(job, weights, 1.0, ReviewAgent())
print(insight.status)
for line in insight.why_lines:
    print(' -', line.text)
"
```

### Tests

```bash
uv run --extra dev pytest tests/unit/m3_production_delay/review tests/unit/m3_production_delay/review_agent -q
```

Offline, no network, no database. The judge is always a scripted callable.
Fixtures come from the rule engine's own test builders (loaded by path in
`conftest.py`, never re-declared) and from a recorded real Section 1 output.

Regenerate the recorded fixture or the evaluation set after a deliberate
change:

```bash
uv run python modules/m3_production_delay/review/fixtures/build_fixture.py
uv run python modules/m3_production_delay/review/eval/build_judge_cases.py
```

### Scoring a real judge

`eval/judge_cases.json` holds 20 cases over four real evidence packs — 10
supported, and 10 unsupported split across `fabricated_cause`,
`wrong_number`, `non_fired_signal` and `omitted_signal`. Every supported case
is verified offline to pass this package's own validators, so the ground truth
cannot disagree with the deterministic layer.

```bash
M3_REVIEW_JUDGE_EVAL=1 LLM_PROVIDER=openai LLM_MODEL_ID=<verified-id> \
  uv run python modules/m3_production_delay/review/eval/run_judge_eval.py
```

It prints per-case results, a per-category breakdown, and precision / recall /
F1 / pinpoint accuracy for the task "withhold approval from an insight that
should not ship". Without the env flag it exits immediately, so no test run
can call a paid API by accident.

---

## Files

| File | Purpose |
|---|---|
| `schemas.py` | Frozen dataclasses with `__post_init__` validation and `to_dict()`/`from_dict()`. No non-finite number is constructible. Re-exports the agent's verdict types. |
| `evidence.py` | `build_evidence(job, weights, threshold)` — the only reader of the Risk Engine's output shape. |
| `composer.py` | Template lines, one per fired weighted signal, ordered by contribution. No LLM. |
| `validators.py` | Every deterministic check, run before the judge. |
| `pipeline.py` | `review_job` / `review_jobs`, plus the tenant batch entry point and CLI. |
| `publish.py` | `customElements` merge and `audit_logs` writeback. |
| `fixtures/` | A recorded real Section 1 output and the script that regenerates it. |
| `eval/` | The judge evaluation set, its builder, and the opt-in runner. |

And the agent, in `llm_agents/review_agent/`:

| File | Purpose |
|---|---|
| `config.py` | `ReviewAgentConfig` + cached singleton — token budget, attempt budget, temperature/seed, enabled flag. |
| `models.py` | `JudgeVerdict` / `JudgeLineVerdict` — the agent's own output contract, capped and validated. |
| `resolver.py` | `ReviewAgent.judge()` — prompt assembly, the one call, strict verdict parsing. |
| `tracing.py` | Opt-in, two-switch execution trace. |
| `exceptions.py` | `ReviewAgentError`, `ReviewConfigError`, `JudgeResponseError`. |
