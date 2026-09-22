// Mirrors modules/m3_production_delay/review/schemas.py `to_dict()` output
// field-for-field — this is the exact JSON GET /api/{tenant}/delay-insights
// returns (each row's `insight`), and what
// manufacturing_orders.custom_elements.ai_delay_insight holds on disk. Kept
// in lockstep with schemas.py rather than loosened, so this file doubles as
// a check that the demo data below is shaped like the real thing.

// The real string values from modules/m3_production_delay/review/schemas.py
// (SIGNAL_TIME_OVERRUN etc.) — confirmed against a live pipeline run, not
// guessed: they carry the rule engine's "_ratio" suffix, e.g.
// "time_overrun_ratio", not "time_overrun".
export type SignalKey =
  | "time_overrun_ratio"
  | "operator_pace_ratio"
  | "material_shortfall_ratio"
  | "supplier_reliability";

export type InsightStatus =
  | "approved"
  | "approved_with_warnings"
  | "fallback_template"
  | "rejected"
  | "suppressed_not_scorable";

export interface InsightLine {
  index: number;
  signal_key: SignalKey;
  scope: "operation" | "job";
  headline: string;
  detail: string;
  delta: string | null;
  operation_id: string | null;
  contribution: number | null;
}

export interface ComponentEvidence {
  component_id: string;
  name: string | null;
  required_quantity: number | null;
  available_quantity: number | null;
  shortfall_quantity: number | null;
  vendor_name?: string | null;
}

export interface JudgeVerdict {
  approved: boolean;
  reason: string;
}

export interface ValidatedInsight {
  job_id: string;
  status: InsightStatus;
  overrun_hours: number | null;
  // Unbounded weighted mean of raw ratios (each signal is 1.0 at exactly
  // "on schedule" and climbs past it) — NOT a 0..1 probability. Confirmed
  // against a live pipeline run: a job 88% over its planned duration with
  // three short components scored 2.23, comfortably clearing a
  // delay_threshold of 1.0. Compare against `delay_threshold`, never against
  // a fixed 0..1 scale.
  risk_score: number | null;
  is_delayed: boolean | null;
  summary_basis: string;
  delay_threshold: number;
  why_lines: InsightLine[];
  material_overrun: ComponentEvidence[];
  model_version: string;
  generated_at: string | null;
}

// --- everything below is presentation-only scaffolding around the MO page,
// not part of M3's contract (the module produces the insight, not the MO
// form itself) ---

export interface ComponentRow {
  product: string;
  availability: "Available" | "Short" | "Not Available";
  toConsume: number;
}

export interface ActivityEntry {
  actor: string;
  initials: string;
  timestamp: string;
  title: string;
}

export type MoStage = "draft" | "confirmed" | "done";

export interface ManufacturingOrder {
  reference: string;
  product: string;
  quantity: number;
  bom: string;
  scheduledDate: string;
  stage: MoStage;
  components: ComponentRow[];
  activity: ActivityEntry[];
  insight: ValidatedInsight;
}

// --- the "Create MO" flow: POST /api/{tenant}/demo-manufacturing-orders
// (scripts/m3_demo_api.py only — the real Configurator API has no MO-create
// route, see frontend/README.md) ---

export interface CreateMoComponentInput {
  name: string;
  required_quantity: number;
  available_quantity: number;
}

export interface CreateMoInput {
  product: string;
  quantity: number;
  operation_name: string;
  expected_duration_hours: number;
  actual_duration_hours: number;
  threshold: number;
  components: CreateMoComponentInput[];
}

export interface CreateMoResponse {
  tenant: string;
  job_id: string;
  insight: ValidatedInsight;
}

// --- the "Admin: Configure Weights" flow: POST /api/{tenant}/demo-weights
// (scripts/m3_demo_api.py only). Mirrors
// modules/m3_production_delay/llm_agents/weight_agent/models.py's
// WeightResolution.to_json_dict() field-for-field, confirmed against a live
// call, not guessed. ---

// The Weight Agent's OWN five signal names (WeightName SIGNAL_ORDER) — NOT
// the same strings as SignalKey above, which are the rule engine's `_ratio`
// vocabulary. weights_bp_to_risk_weights() is the one place that translates
// between them (modules/m3_production_delay/rule_engine/elements.py).
export type WeightSignalName =
  | "time_overrun"
  | "operator_skill"
  | "seasonality"
  | "material_availability"
  | "supplier_reliability";

export const WEIGHT_SIGNAL_ORDER: WeightSignalName[] = [
  "time_overrun",
  "operator_skill",
  "seasonality",
  "material_availability",
  "supplier_reliability",
];

// config.py DEFAULT_BOUNDS_BP / DEFAULT_PRIOR_BP — confirmed by reading that
// file, used to pre-fill the form and keep the sliders inside bounds the
// server will actually accept.
export const WEIGHT_BOUNDS_BP: Record<WeightSignalName, { min: number; max: number }> = {
  time_overrun: { min: 3000, max: 5000 },
  operator_skill: { min: 2000, max: 4500 },
  seasonality: { min: 500, max: 2000 },
  material_availability: { min: 500, max: 2500 },
  supplier_reliability: { min: 300, max: 1500 },
};

export const WEIGHT_PRIOR_BP: Record<WeightSignalName, number> = {
  time_overrun: 4000,
  operator_skill: 3500,
  seasonality: 1000,
  material_availability: 1000,
  supplier_reliability: 500,
};

export type WeightSource = "configured" | "historically_fitted" | "blended" | "llm_adjusted_prior" | "prior";
export type WeightStatus = "active" | "recommendation";

export interface HistoryAssessment {
  sufficient: boolean;
  admissibility: "inadmissible" | "usable_weak" | "sufficient" | "preferred";
  reasons: string[];
  lambda_bp: number;
}

export interface WeightResolution {
  tenant_id: string;
  source: WeightSource;
  status: WeightStatus;
  weights_bp: Record<WeightSignalName, number>;
  available_signals: WeightSignalName[];
  excluded_signals: { signal: WeightSignalName; reason: string }[];
  history_assessment: HistoryAssessment;
  prior_bp: Record<WeightSignalName, number>;
  adjustments_bp: Record<WeightSignalName, number>;
  bounds_bp: Record<WeightSignalName, { min: number; max: number }>;
  confidence: number;
  evidence: string[];
  fallback_reasons: string[];
  requires_admin_approval: boolean;
  generated_at: string;
}

export interface ResolveWeightsInput {
  configured_bp: Record<WeightSignalName, number> | null;
  job_reference: string;
  threshold: number;
}

export interface ResolveWeightsResponse {
  tenant: string;
  weight_resolution: WeightResolution;
  risk_weights: Record<string, number>;
  job_id: string;
  insight: ValidatedInsight;
}
