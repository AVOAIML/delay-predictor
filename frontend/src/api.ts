import type {
  CreateMoInput,
  CreateMoResponse,
  ManufacturingOrder,
  ManufacturingOrderOptionsResponse,
  ManufacturingOrdersResponse,
  ProductOption,
  ResolveWeightsInput,
  ResolveWeightsResponse,
  ValidatedInsight,
} from "./types";
import { SCENARIOS } from "./data/scenarios";

// Talks to the real Configurator API when one is configured, otherwise
// serves the bundled fixtures so this demo runs standalone with no
// database, auth or backend at all (`npm install && npm run dev`).
//
// The real endpoint is GET /api/{tenant}/delay-insights?job=<reference>
// (services/configurator/app.py `delay_insights`), which requires a bearer
// token and an `x-tenant-slug` header (see security.py
// `authenticate_request`). Set VITE_API_BASE + VITE_TENANT + VITE_TOKEN in
// frontend/.env.local to point this demo at a live stack instead of mocks.
const API_BASE = import.meta.env.VITE_API_BASE as string | undefined;
const TENANT = (import.meta.env.VITE_TENANT as string | undefined) ?? "demo";
const TOKEN = import.meta.env.VITE_TOKEN as string | undefined;

export async function fetchDelayInsight(jobReference: string): Promise<ValidatedInsight> {
  if (API_BASE) {
    const res = await fetch(
      `${API_BASE}/api/${encodeURIComponent(TENANT)}/delay-insights?job=${encodeURIComponent(jobReference)}`,
      {
        headers: {
          "x-tenant-slug": TENANT,
          ...(TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {}),
        },
      },
    );
    if (!res.ok) {
      throw new Error(`delay-insights ${res.status}: ${await res.text()}`);
    }
    const body = (await res.json()) as { insights: { job_id: string; insight: ValidatedInsight }[] };
    const row = body.insights[0];
    if (!row) throw new Error(`no cached delay insight for ${jobReference}`);
    return row.insight;
  }

  const fixture = SCENARIOS.find((order) => order.reference === jobReference);
  if (!fixture?.insight) throw new Error(`no demo insight for ${jobReference}`);
  // Mimic network latency so loading states are visible in the demo.
  await new Promise((resolve) => setTimeout(resolve, 150));
  return fixture.insight;
}

export const usingLiveApi = Boolean(API_BASE);

export async function fetchManufacturingOrders(): Promise<ManufacturingOrder[]> {
  if (!API_BASE) return SCENARIOS;
  const res = await fetch(
    `${API_BASE}/api/${encodeURIComponent(TENANT)}/demo-manufacturing-orders`,
  );
  if (!res.ok) {
    throw new Error(`manufacturing-orders ${res.status}: ${await res.text()}`);
  }
  const body = (await res.json()) as ManufacturingOrdersResponse;
  return body.orders;
}

export async function fetchManufacturingOrderOptions(): Promise<ProductOption[]> {
  if (!API_BASE) return [];
  const res = await fetch(
    `${API_BASE}/api/${encodeURIComponent(TENANT)}/demo-manufacturing-order-options`,
  );
  if (!res.ok) {
    throw new Error(`manufacturing-order-options ${res.status}: ${await res.text()}`);
  }
  const body = (await res.json()) as ManufacturingOrderOptionsResponse;
  return body.products;
}

// Real Configurator API has no equivalent route — MO creation belongs to the
// main MaXXFlow product, not this AI/ML repo. This only works against
// scripts/m3_demo_api.py's POST /api/{tenant}/demo-manufacturing-orders,
// which inserts the job into real MRP tables and runs the actual
// m3_production_delay.review pipeline against it before returning.
export async function createManufacturingOrder(input: CreateMoInput): Promise<CreateMoResponse> {
  if (!API_BASE) {
    throw new Error(
      "Creating a test MO needs the live demo bridge — set VITE_API_BASE in frontend/.env.local " +
        "(see frontend/README.md) and run `uv run uvicorn scripts.m3_demo_api:app --port 8010`.",
    );
  }
  const res = await fetch(`${API_BASE}/api/${encodeURIComponent(TENANT)}/demo-manufacturing-orders`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) {
    throw new Error(`demo-manufacturing-orders ${res.status}: ${await res.text()}`);
  }
  return (await res.json()) as CreateMoResponse;
}

// Runs the REAL Weight Agent (ProductionDelayOrchestrator.resolve_weights)
// against `configured_bp` (or cold-start when null) and re-scores
// `job_reference` with exactly the weights it resolved. Same local-bridge-only
// caveat as createManufacturingOrder above — no equivalent route exists on
// the real, authenticated Configurator API.
export async function resolveWeights(input: ResolveWeightsInput): Promise<ResolveWeightsResponse> {
  if (!API_BASE) {
    throw new Error(
      "Testing the Weight Agent needs the live demo bridge — set VITE_API_BASE in " +
        "frontend/.env.local (see frontend/README.md) and run " +
        "`uv run uvicorn scripts.m3_demo_api:app --port 8010`.",
    );
  }
  const res = await fetch(`${API_BASE}/api/${encodeURIComponent(TENANT)}/demo-weights`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) {
    throw new Error(`demo-weights ${res.status}: ${await res.text()}`);
  }
  return (await res.json()) as ResolveWeightsResponse;
}
