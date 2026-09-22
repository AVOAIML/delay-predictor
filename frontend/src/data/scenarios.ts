import type { ManufacturingOrder } from "../types";

// Three fixtures standing in for `GET /api/{tenant}/delay-insights`. The
// `insight` block on each is shaped exactly like
// `ValidatedInsight.to_dict()` (see src/types.ts) and its headline/detail
// strings are copied verbatim from
// modules/m3_production_delay/review/composer.py's HEADLINES table and
// template functions — this is what the Review Agent's deterministic
// composer actually renders for a fired, weighted signal, not invented copy.
//
// The three orders exercise three different fired signals so switching
// between them demonstrates M3's actual range of behaviour: a pure
// time-overrun cause (high risk), a blended operator-pace + material cause
// (medium risk), and a healthy order with nothing fired (low risk, no
// why_lines — schemas.py allows that: material_overrun is independent of
// is_delayed, but why_lines can be empty when no weighted signal fired).

export const SCENARIOS: ManufacturingOrder[] = [
  {
    reference: "WH/MO/00142",
    product: "Wooden Table",
    quantity: 1.0,
    bom: "Wooden Table BoM",
    scheduledDate: "25/10/2025",
    stage: "confirmed",
    components: [
      { product: "Table Top", availability: "Available", toConsume: 1 },
      { product: "Table Leg", availability: "Available", toConsume: 4 },
      { product: "Screws", availability: "Available", toConsume: 8 },
    ],
    activity: [
      {
        actor: "Sam Smith",
        initials: "SS",
        timestamp: "26 Aug 2025, 8:30 AM",
        title: "Manufacturing Order Details Updated",
      },
      {
        actor: "Sam Smith",
        initials: "SS",
        timestamp: "26 Aug 2025, 8:30 AM",
        title: "Manufacturing Order Created",
      },
    ],
    // These exact numbers are not invented — they're what
    // `python -m m3_production_delay.review --tenant demo --threshold 1.0
    // --job "WH/MO/00142"` actually returned against this same job seeded by
    // scripts/m3_demo_seed.py (see scripts/m3_demo_api.py for how to read it
    // back live instead of from this fixture). `status` is
    // "fallback_template" because the local stub LLM doesn't return JSON, so
    // the Review Agent's judge call is rejected and the deterministic
    // template lines ship unjudged — expected behaviour locally, not a bug.
    // The operation reads as "INDEPENDENT · <id-prefix>" rather than
    // "Assemble Table" because build_job_rollups() never carries
    // `operation_name` through from the DB read into the per-operation dict
    // the evidence layer sees (verified by reading
    // modules/m3_production_delay/rule_engine/rollup.py) — a real gap in the
    // module, not a display bug.
    insight: {
      job_id: "WH/MO/00142",
      status: "fallback_template",
      overrun_hours: 3.5,
      risk_score: 2.2285714285714286,
      is_delayed: true,
      summary_basis: "worst_operation",
      delay_threshold: 1.0,
      why_lines: [
        {
          index: 0,
          signal_key: "time_overrun_ratio",
          scope: "operation",
          headline: "Actual time logged has exceeded expected duration",
          detail: "INDEPENDENT · 6e117fe4 · 7.5 hrs logged of 4.0 planned",
          delta: "+3.5 hrs",
          operation_id: "6e117fe4-203b-488a-8467-9786df88bc55",
          contribution: 0.673076923076923,
        },
        {
          index: 1,
          signal_key: "material_shortfall_ratio",
          scope: "job",
          headline: "Required material is short in the warehouse",
          detail: "WH/MO/00142 · Steel Rod – 2 short; Screws – 5 short; Nuts – 12 short",
          delta: null,
          operation_id: null,
          contribution: 0.3269230769230769,
        },
      ],
      material_overrun: [
        {
          component_id: "COMP-STEEL-ROD",
          name: "Steel Rod",
          required_quantity: 10,
          available_quantity: 8,
          shortfall_quantity: 2,
        },
        {
          component_id: "COMP-SCREWS",
          name: "Screws",
          required_quantity: 40,
          available_quantity: 35,
          shortfall_quantity: 5,
        },
        {
          component_id: "COMP-NUTS",
          name: "Nuts",
          required_quantity: 60,
          available_quantity: 48,
          shortfall_quantity: 12,
        },
      ],
      model_version: "m3-review-v1",
      generated_at: "2026-09-21T21:00:52.085860+10:00",
    },
  },
  {
    reference: "WH/MO/00151",
    product: "Steel Cabinet",
    quantity: 3.0,
    bom: "Steel Cabinet BoM",
    scheduledDate: "02/11/2025",
    stage: "confirmed",
    components: [
      { product: "Cabinet Frame", availability: "Available", toConsume: 3 },
      { product: "Door Hinge", availability: "Short", toConsume: 12 },
      { product: "Lock Set", availability: "Available", toConsume: 3 },
    ],
    activity: [
      {
        actor: "Priya Nair",
        initials: "PN",
        timestamp: "18 Sep 2025, 2:05 PM",
        title: "Manufacturing Order Confirmed",
      },
      {
        actor: "Priya Nair",
        initials: "PN",
        timestamp: "18 Sep 2025, 9:12 AM",
        title: "Manufacturing Order Created",
      },
    ],
    insight: {
      job_id: "WH/MO/00151",
      status: "approved_with_warnings",
      overrun_hours: 1.2,
      risk_score: 0.55,
      is_delayed: true,
      summary_basis: "worst_operation",
      delay_threshold: 0.5,
      why_lines: [
        {
          index: 0,
          signal_key: "operator_pace_ratio",
          scope: "operation",
          headline: "Assigned operator has a history of overrunning",
          detail: "Weld Frame · operator pace 1.35× planned over recent completed jobs",
          delta: null,
          operation_id: "OP-WELD-02",
          contribution: 0.6,
        },
        {
          index: 1,
          signal_key: "material_shortfall_ratio",
          scope: "job",
          headline: "Required material is short in the warehouse",
          detail: "WH/MO/00151 · Door Hinge – 9 short",
          delta: null,
          operation_id: null,
          contribution: 0.4,
        },
      ],
      material_overrun: [
        {
          component_id: "COMP-DOOR-HINGE",
          name: "Door Hinge",
          required_quantity: 36,
          available_quantity: 27,
          shortfall_quantity: 9,
        },
      ],
      model_version: "m3_production_delay@2026.03.1",
      generated_at: "2025-09-18T09:12:00Z",
    },
  },
  {
    reference: "WH/MO/00163",
    product: "Office Chair",
    quantity: 5.0,
    bom: "Office Chair BoM",
    scheduledDate: "05/11/2025",
    stage: "draft",
    components: [
      { product: "Chair Base", availability: "Available", toConsume: 5 },
      { product: "Gas Lift", availability: "Available", toConsume: 5 },
      { product: "Armrest", availability: "Available", toConsume: 10 },
    ],
    activity: [
      {
        actor: "Priya Nair",
        initials: "PN",
        timestamp: "20 Sep 2025, 11:40 AM",
        title: "Manufacturing Order Created",
      },
    ],
    insight: {
      job_id: "WH/MO/00163",
      status: "approved",
      overrun_hours: 0,
      risk_score: 0.12,
      is_delayed: false,
      summary_basis: "worst_operation",
      delay_threshold: 0.5,
      why_lines: [],
      material_overrun: [],
      model_version: "m3_production_delay@2026.03.1",
      generated_at: "2025-09-20T11:40:00Z",
    },
  },
];
