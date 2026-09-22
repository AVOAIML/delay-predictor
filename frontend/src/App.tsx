import { useEffect, useState } from "react";
import { SCENARIOS } from "./data/scenarios";
import { fetchDelayInsight, usingLiveApi } from "./api";
import type { ManufacturingOrder, ValidatedInsight } from "./types";
import { TopBar } from "./components/TopBar";
import { MoHeader } from "./components/MoHeader";
import { OrderForm } from "./components/OrderForm";
import { DelayRiskCard } from "./components/DelayRiskCard";
import { MaterialOverrunCard } from "./components/MaterialOverrunCard";
import { ActivityTimeline } from "./components/ActivityTimeline";
import { CreateMoPanel } from "./components/CreateMoPanel";
import { WeightsAdminPanel } from "./components/WeightsAdminPanel";

type View = "mo" | "create-mo" | "weights";

export default function App() {
  // Seeded with the three bundled fixtures; a successful Create MO submit
  // appends the real, freshly-scored job to this same list.
  const [orders, setOrders] = useState<ManufacturingOrder[]>(SCENARIOS);
  const [reference, setReference] = useState(SCENARIOS[0].reference);
  const [view, setView] = useState<View>("mo");
  const order = orders.find((o) => o.reference === reference) ?? orders[0];

  // Re-fetches through api.ts on every scenario switch, exactly like the
  // real MO page would call GET /api/{tenant}/delay-insights?job=<ref> — in
  // mock mode this just resolves the bundled fixture, but the data path is
  // the same one a live backend would take.
  const [insight, setInsight] = useState<ValidatedInsight>(order.insight);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchDelayInsight(reference)
      .then((result) => {
        if (!cancelled) setInsight(result);
      })
      .catch(() => {
        if (!cancelled) setInsight(order.insight);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reference]);

  const handleCreated = (newOrder: ManufacturingOrder) => {
    setOrders((prev) => [...prev, newOrder]);
    setInsight(newOrder.insight);
    setReference(newOrder.reference);
    setView("mo");
  };

  // The Weight Agent panel re-scores an EXISTING job in place — update that
  // order's stored insight so returning to its normal MO view reflects the
  // new weights, and update the live `insight` state directly too (no
  // `reference` change happens here to re-trigger the fetch effect above).
  const handleRescored = (jobReference: string, newInsight: ValidatedInsight) => {
    setOrders((prev) =>
      prev.map((o) => (o.reference === jobReference ? { ...o, insight: newInsight } : o)),
    );
    if (jobReference === reference) setInsight(newInsight);
  };

  return (
    <div className="page">
      <TopBar
        orders={orders}
        selected={reference}
        onSelect={setReference}
        onCreateClick={usingLiveApi ? () => setView("create-mo") : undefined}
        onWeightsClick={usingLiveApi ? () => setView("weights") : undefined}
      />
      {!usingLiveApi && (
        <div className="demo-banner">
          Demo mode — serving bundled M3 fixtures, not a live Configurator API. Set{" "}
          <code>VITE_API_BASE</code> in <code>frontend/.env.local</code> to point this at a real
          stack (see frontend/README.md) — that also unlocks "+ New MO" and "Weights (Admin)".
        </div>
      )}
      <main className={view === "mo" ? "layout" : "layout layout--single"}>
        <div className="layout__main">
          {view === "create-mo" && (
            <CreateMoPanel onCreated={handleCreated} onCancel={() => setView("mo")} />
          )}
          {view === "weights" && (
            <WeightsAdminPanel
              orders={orders}
              defaultJobReference={reference}
              onRescored={handleRescored}
              onClose={() => setView("mo")}
            />
          )}
          {view === "mo" && (
            <>
              <MoHeader order={order} />
              <OrderForm order={order} />
            </>
          )}
        </div>
        {view === "mo" && (
          <aside className="layout__side">
            <DelayRiskCard insight={loading ? order.insight : insight} />
            <MaterialOverrunCard components={(loading ? order.insight : insight).material_overrun} />
            <ActivityTimeline entries={order.activity} />
          </aside>
        )}
      </main>
    </div>
  );
}
