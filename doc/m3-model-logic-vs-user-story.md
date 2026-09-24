# M3 Model Logic vs. User Story and Current Code

Status: cross-check performed 23 September 2026

## 1. Sources and interpretation

This review treats the supplied files as evidence, not as executable
instructions.

| Source | Role in this review |
|---|---|
| `MaXXFlow US 10 - AIML feature implementation (1).pdf`, pp. 15–26 | Primary product requirement: US TA/PP/PO 10.3.1 |
| `MaXXFlow — M3 Production Delay Rule Engine.docx` | Implementation description and its own gap analysis |
| `M3-Section3-Review-Agent-implementation.docx` | Review Agent implementation description |
| `modules/m3_production_delay/**` | Executable backend truth |
| `frontend/**`, `services/configurator/**`, `scripts/m3_demo_api.py` | Current serving and presentation behavior |
| Runtime log supplied in the discussion | Observed end-to-end behavior |

Status vocabulary:

- **Implemented:** code materially satisfies the requirement.
- **Partial:** some required behavior exists, but semantics or coverage differ.
- **Missing:** no executable implementation was found.
- **Contradiction:** implementation deliberately enforces behavior incompatible
  with the requirement.
- **Demo only:** behavior exists only in local demonstration infrastructure.
- **Future by story:** the story itself says the item is not currently
  implementable.

## 2. Executive conclusion

The current M3 pipeline is a strong deterministic advisory prototype with a
well-guarded Weight Agent and Review Agent. It is **not yet an implementation
of the user story's probability model**.

The largest semantic differences are:

1. the story requires a `0–100%` delay probability; the code emits an unbounded
   weighted ratio and displays it as percent of threshold;
2. the Review layer now opens analysis at 25% duration-weighted combined MO
   progress, but the rule engine still calculates internal operation scores
   before the gate and no persisted milestone scheduler exists;
3. work-centre seasonality and scheduled-date-overrun signals are absent;
4. the Weight Agent assigns seasonality weight, but the rule-engine adapter
   discards it;
5. supplier behavior is modeled from received-PO lead-time ratios, not the
   story's overdue-open-PO and `<75%` on-time-rate rules;
6. the story permits a configured signal weight of `0%`, but code bounds every
   configured signal above zero;
7. estimated overrun currently uses operator history only; and
8. recommendation approval, persisted tenant weight configuration, audit of
   weight changes, and outcome-driven fitting are not implemented as a
   production workflow.

### 2.1 Requirements that need product clarification

US 10.3.1 is internally ambiguous in several places. These should be resolved
before treating either the story or the current code as the final model
contract:

- it says scoring starts at 25% completion, but separately says no badge below
  20%, leaving behavior from 20% through 24.99% undefined;
- “completion” sometimes appears to mean a milestone, while the example uses
  percentage of the expected time budget; these are different measurements;
- it says no historical training dependency is required, but also says the
  model compares against completed jobs, operator history, vendor history, and
  at least two years of work-centre seasonality;
- it calls the output a probability, but provides ratio and rule inputs without
  a calibration method that maps them to a probability;
- it lists concurrent work-centre load and BOM complexity as training inputs
  without defining their formulas, thresholds, or configurable weights;
- it says no retraining is required, while also saying the Weight Configuration
  Agent may update recommendations as outcomes accumulate; and
- it requires both a default 71% alert threshold and Tenant Admin configuration,
  but does not define whether the configured value is a probability cutoff, a
  risk-index cutoff, or a badge boundary.

The current code resolves these ambiguities in particular ways, but those code
choices should not be mistaken for signed-off product decisions.

## 3. Requirement traceability matrix

### 3.1 Scoring semantics and triggering

| User-story requirement | Current implementation | Status | Evidence / consequence |
|---|---|---|---|
| Start analysis once combined Manufacturing Order progress reaches 25%. | `elements.py` calculates duration-weighted MO progress as `Σ(expected duration × WO quantity progress) / Σ(expected duration)`; `review/evidence.py` suppresses user-facing analysis below `0.25`. | **Implemented for presentation gate** | Internal operation signals are still calculated before the gate. |
| No badge or score below 20% completion. | Review output remains suppressed below the stricter 25% combined-MO gate. | **Implemented for presentation gate** | Internal operation calculations still exist for diagnostics. |
| Re-score at each 25% milestone, not continuously. | Batch/API invocation triggers scoring. No milestone scheduler or persisted milestone state was found. | **Missing** | The demo scores immediately after job creation and on manual weight rescoring. |
| Output a probability from 0% to 100%. | Weighted mean of raw ratios; unbounded and not calibrated. | **Contradiction** | Observed score `1.9975` becomes `200% of delay threshold`. |
| Default alert threshold 71%, tenant configurable. | Rule-engine placeholder threshold is `1.0`; production batch endpoint requires an explicit numeric threshold. | **Contradiction** | Threshold has index units, not probability units. |
| Badge colors follow probability bands. | Frontend uses delayed/not-delayed plus `>=150% of threshold` for High Risk. | **Contradiction** | This is a presentation band over threshold, not the story's probability bands. |
| Score is advisory and does not block the Work Order. | Pipeline writes an advisory JSON payload and does not alter workflow state. | **Implemented** | Cached under `ai_delay_insight`. |

### 3.2 Delay signals

| User-story signal or rule | Current implementation | Status | Gap |
|---|---|---|---|
| Time Overrun Ratio = actual duration / expected duration. | Implemented as actual duration divided by the expected duration for the completed quantity fraction. | **Partial** | The progress-normalized formula detects pace overrun, but differs from the story's plain ratio wording. |
| Time overrun is strongest. | Default Weight Agent prior gives it 40%, the largest single weight. | **Implemented** | Final LLM/configured weights may change it within 30–50%. |
| Operator risk from last 10 completed WOs; high risk above 1.20. | Continuous `operator_pace_ratio` uses the last 10 completed WOs; explanation fires above 1.20. | **Implemented** | Composite score uses the continuous ratio rather than a discrete skill tier. |
| Operator identity not stored; only skill tier used. | Raw IDs are HMAC-pseudonymized, but stable per-operator tokens and continuous histories remain in the rollup. | **Partial** | Privacy protection exists, but this is not “skill tier only.” |
| Scheduled-date overrun strengthens with days late. | Scheduled fields are read by the DAL but not used in `elements.py`. | **Missing** | No days-overdue feature or weight. |
| Cascading predecessor risk above 1.20. | CPM identifies zero-float critical edges; an overrun above 1.20 propagates to the not-started dependent, raises its score, can mark it delayed, and produces a visible reason. | **Implemented** | Independent operations and non-critical dependency branches are deliberately excluded. |
| Work-centre monthly low throughput above 1.20 with at least two years of history. | No monthly work-centre aggregation exists. | **Missing** | Weight Agent seasonality does not create a scoring signal. |
| Concurrent Work Order count on the same Work Center. | Work centers are read, but concurrent load is not calculated or scored. | **Missing** | No capacity/congestion feature. |
| BOM component/operation counts as complexity context. | Not calculated in rollup or elements. | **Missing** | No BOM complexity signal. |
| Material signal reflects consumption/wastage exceeding plan and unavailable stock. | Current signal sums `required / available` for any BOM component with `available < required`. | **Partial** | It detects static shortage, not consumption overrun or wastage-driven shortfall. |
| Open overdue PO with no GRN, days overdue, and vendor on-time rate below 75%. | Current vendor ratio averages received-PO lead time; open POs without a GRN are excluded. | **Contradiction** | No overdue-open-PO exposure and no 75% on-time-rate calculation. |
| Wastage overrun. | No planned/actual wastage fields are read. | **Future by story** | The story explicitly says operation-level wastage is not yet implemented. |
| Combine all active signals using tenant weights. | Four supported operation signals form a dynamically renormalized base score; critical-path cascade is a deterministic overlay. | **Partial** | Seasonality still has no executable scoring signal; cascade is rule-driven rather than tenant-weighted. |

### 3.3 Weight Configuration Agent

| User-story requirement | Current implementation | Status | Evidence / consequence |
|---|---|---|---|
| Defaults: 40/35/10/10/5. | Domain prior exactly matches. | **Implemented** | Stored as 10,000 basis points. |
| Admin may configure each signal from 0% to 100%; total must equal 100%. | Total must equal 100%, but each signal must remain within narrower bounds: 30–50, 20–45, 5–20, 5–25, 3–15%. | **Contradiction** | A signal cannot be configured to 0%, although the story requires disabling it this way. |
| Zero-weight signal is excluded from score and explanation. | Review hides zero-weight signals, but standard configured weights cannot reach zero under current bounds. Availability masking can force zero. | **Partial** | The behavior exists structurally but is inaccessible through normal configuration. |
| Agent recommends initial weights using industry/MRP context. | Tenant description → validated profile → LLM sum-to-zero adjustment of prior. | **Implemented** | Exact adjustment magnitude is LLM-selected; direction and final bounds are controlled. |
| Recommendation shown for Admin acceptance/modification before application. | `requires_admin_approval=True` is emitted for recommendations. No production persistence/accept endpoint was found. | **Partial** | The contract represents approval need, but does not complete the workflow. |
| Manual configuration available under Settings. | A local demo panel submits explicit weights and immediately re-scores. | **Demo only** | It is not the authenticated production Settings workflow. |
| Reset to Defaults. | Frontend initializes to defaults, but no explicit production reset action/API was found. | **Missing** | Demo state can be manually restored only by re-entering values/reloading. |
| Weight changes are audited with actor and timestamp. | Weight resolutions are logged, but no DB persistence/audit workflow for accepted weight changes was found. | **Missing** | Review-result audit rows are separate from weight-change audit. |
| Accepted changes affect the next milestone score. | Configured weights can affect an immediate request. No persisted accepted configuration or milestone scheduler was found. | **Partial** | Demonstrated computationally, not operationally. |
| Recommendations may evolve as outcomes accumulate. | Historical blend interfaces and formulas exist. Shipped provider always returns no fitted artifact. | **Partial** | No training/fitting pipeline, artifact store, or production provider. |

### 3.4 Estimated overrun

| User-story expectation | Current implementation | Status | Gap |
|---|---|---|---|
| Show estimated overrun hours for the active job. | Per-operation estimate exists and the job summary takes the maximum scorable operation value. | **Partial** | “Total” is not a critical-path/schedule total. |
| Estimate reflects current job progress. | Quantity-based estimate-at-completion extrapolates actual time from completed quantity. | **Implemented** | Operator history remains the fallback when quantity progress is unavailable. |
| Actual logged overrun is reflected. | Actual logged time drives the quantity-based estimate when progress exists. | **Implemented** | Without quantity progress, the estimate falls back to operator history. |
| Explanation accurately states the estimate basis. | Evidence reports `quantity`, `operator_pace`, or `none` using the same branch conditions as the engine. | **Implemented** | — |

### 3.5 Explanation and Review Agent

| User-story requirement | Current implementation | Status | Evidence / consequence |
|---|---|---|---|
| Explain only signals that fired and are supported by ERP/MRP evidence. | Deterministic composer emits fired score causes, including a validated critical-path cascade; validators check evidence and numbers. | **Implemented** | Stronger structural protection than free-form generation. |
| GenAI generates plain-language explanation. | Templates generate the text; LLM only judges it. | **Partial / safer deviation** | Functional explanation exists without allowing the LLM to invent text or numbers. |
| Review Agent validates every explanation. | It judges after deterministic validation unless the insight is rejected/not scorable. | **Implemented** | Normal candidate text reaches one or two judge attempts. |
| Unsupported/omitted explanation is rejected and regenerated. | Unsupported lines may be dropped and re-judged. If judgment still fails, the full template draft is published as `fallback_template`. | **Partial** | Failed judgment does not withhold the explanation; it publishes an audited unjudged fallback. |
| Investigation Agent appears for complex/unclear cases. | No Investigation Agent implementation was found. | **Missing** | No complexity classifier or conditional investigation section. |
| Material Overrun list shows only affected components. | Components with `available < required` are deduplicated and published. | **Partial** | Static shortage semantics differ from consumption-overrun semantics. |
| Material list remains visible below the delay threshold. | Material evidence is included for all review statuses. | **Implemented** | Independent of explanation status. |

### 3.6 UI, operations, and security

| User-story requirement | Current implementation | Status | Evidence / consequence |
|---|---|---|---|
| Read-only AI Insights panel. | Frontend cards are display-only. | **Implemented** | Demo frontend supports display. |
| Risk, overrun, reasons, and material components appear together. | `ValidatedInsight` and the demo UI contain these sections. | **Implemented** | Risk percent semantics differ from the story. |
| Gracefully hide risk when scoring service is unavailable. | API returns an error; demo UI shows request errors. | **Missing** | No verified production hide/degrade behavior. |
| Tenant-scoped scoring and reads. | DAL uses tenant-scoped access; authenticated Configurator route checks path tenant against auth context. | **Implemented** | Local demo bridge is intentionally unauthenticated and local-only. |
| Persist score for later display. | Insight is merged into the MO's `custom_elements`. | **Implemented** | Read API returns cached JSON. |
| Activity Timeline continues below insights. | Outside core model logic; not verified in the production product UI from this repository. | **Not verified** | Demo UI does not prove main-product placement. |

## 4. Cross-check of supplied implementation documents

### 4.1 Rule Engine DOCX

The Rule Engine document closely matches the current formulas and accurately
identifies missing work-centre, live overdue-PO, and wastage features. Its most
important caveats are also visible in code:

- the threshold `1.0` is a placeholder;
- quantity-based overrun uses estimate-at-completion from actual progress;
- seasonality is dropped at the Weight Agent adapter; and
- the composite is an unbounded weighted ratio.

The document's estimate-at-completion description now matches executable
behavior: quantity progress is preferred and operator history is the fallback.

### 4.2 Review Agent DOCX

The Review Agent document is substantially aligned with the current code:

- evidence, composition, deterministic validation, judgment, and publication
  are separate stages;
- candidate text is template-authored;
- the judge cannot edit lines;
- deterministic errors reject before the LLM call; and
- fallback/rejected/not-scorable results are audited.

The security claim should be read precisely: an incorrect judge may approve an
unsupported template line, but it still cannot inject a new number or sentence.
The deterministic validators materially reduce this risk, while
`fallback_template` explicitly allows unjudged template text to ship after
judge failure.

## 5. Severity-ranked implementation gaps

### P0 — semantics must be decided before calling the output a probability

1. Choose one contract:
   - implement and calibrate a true `[0,1]` probability; or
   - rename the UI/API concept to “risk index” and retain percent-of-threshold
     presentation.
2. Define the tenant threshold in the same units as that contract.
3. Align badge bands with the selected contract.

Until this decision is made, a display such as `200%` conflicts directly with
US 10.3.1.

### P0 — implement the scoring trigger correctly

1. Define whether “25% completion” means quantity completion, planned-time
   consumption, or a persisted milestone event.
2. Gate rule-engine scoring, not only Review explanations.
3. Add the next-milestone rescore trigger and prevent continuous/manual calls
   from changing product semantics.

### P1 — complete the specified signal set

1. Scheduled-date overdue days.
2. Work-centre monthly throughput seasonality with the two-year gate.
3. Live overdue-open-PO exposure and vendor on-time rate.
4. Decide whether the deterministic critical-path cascade overlay should later
   become a tenant-configurable weighted signal.
5. Decide whether concurrent work-centre load and BOM complexity are actual
   scoring features.
6. Replace static material shortage with the story's consumption-aware rule,
   or explicitly amend the requirement.

### P1 — repair estimated-overrun semantics

1. Define job-level aggregation using operation dependencies/critical path;
   otherwise rename “Total Estimated Delay” to “Worst Operation Estimated
   Delay.”

### P1 — complete weight administration

1. Reconcile story-required `0%` disabling with non-zero configured bounds.
2. Persist tenant-approved weights.
3. Add accept/modify/reset actions behind Tenant Admin authorization.
4. Audit actor, before/after vector, timestamp, and recommendation provenance.
5. Decide whether bounds are user-story amendments or internal guardrails that
   must be exposed in the acceptance criteria.

### P2 — finish outcome learning and operational guardrails

1. Build the fitted-weight training pipeline and artifact store, or remove the
   route from production claims until it exists.
2. Implement a real `FittedWeightsProvider`.
3. Define outcome labels and evaluation metrics.
4. Add the story's scoring-service-unavailable UI behavior.
5. Decide whether an unapproved `fallback_template` is acceptable or whether
   the story requires withholding/regeneration.
6. Implement or remove the Investigation Agent requirement.

## 6. Suggested acceptance tests

| Test | Expected result |
|---|---|
| Work Order at 19% completion | No persisted score or badge. |
| Work Order crossing 25% milestone | One scoring event; next score waits for the next milestone. |
| Probability contract | Every result is finite and within 0–100%, with calibration evidence. |
| Seasonality unavailable | Seasonality weight is zero and no allocation is silently discarded. |
| Seasonality available with two years of data | Work-centre monthly signal changes the score as specified. |
| Operator pace 1.21 over last 10 completed jobs | Operator signal fires with no raw operator PII exposed. |
| Predecessor ratio 1.21 | Dependent operation behavior matches the agreed weighted/context-only rule. |
| Required component covered | No material-overrun row. |
| Consumption exceeds plan and stock cannot cover | Material signal and component row fire. |
| Open PO overdue, no GRN, vendor on-time rate 70% | Supplier signal fires and strengthens with overdue days. |
| Admin sets one signal to 0% and others total 100% | Save succeeds; signal affects neither score nor explanation. |
| Agent recommendation not accepted | Active weights remain unchanged. |
| Agent recommendation accepted | New weights persist, are audited, and apply at the next milestone. |
| Quantity progress exists but no operator history | Overrun estimate uses quantity EAC and basis reports `quantity`. |
| Judge unavailable | Product behavior follows the explicitly approved fallback policy. |
| Cross-tenant request | Access is denied and no data is read or written. |

## 7. Code map used for verification

| Concern | Current source |
|---|---|
| Tenant reads and pseudonymization | [rule_engine/dal.py](../modules/m3_production_delay/rule_engine/dal.py) |
| Nested job rollup | [rule_engine/rollup.py](../modules/m3_production_delay/rule_engine/rollup.py) |
| Signal formulas, score, threshold, overrun | [rule_engine/elements.py](../modules/m3_production_delay/rule_engine/elements.py) |
| Weight prior, bounds, history policy | [weight_agent/config.py](../modules/m3_production_delay/llm_agents/weight_agent/config.py) |
| Weight source priority | [weight_agent/resolver.py](../modules/m3_production_delay/llm_agents/weight_agent/resolver.py) |
| Historical blending | [weight_agent/blend.py](../modules/m3_production_delay/llm_agents/weight_agent/blend.py) |
| Fitted provider status | [weight_agent/providers.py](../modules/m3_production_delay/llm_agents/weight_agent/providers.py) |
| LLM direction guidance | [prompt/weight_adjustment.txt](../modules/m3_production_delay/prompt/weight_adjustment.txt) |
| Review evidence and 25% gate | [review/evidence.py](../modules/m3_production_delay/review/evidence.py) |
| Review orchestration and fallback | [review/pipeline.py](../modules/m3_production_delay/review/pipeline.py) |
| Insight persistence/audit | [review/publish.py](../modules/m3_production_delay/review/publish.py) |
| Production batch/read endpoints | [services/configurator/app.py](../services/configurator/app.py) |
| Local demo endpoints | [scripts/m3_demo_api.py](../scripts/m3_demo_api.py) |
| Risk percentage presentation | [frontend/src/riskBand.ts](../frontend/src/riskBand.ts) |
