import { useEffect, useState } from "react";
import { SCENARIOS } from "./data/scenarios";
import { fetchManufacturingOrders, usingLiveApi } from "./api";
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
  // Live mode starts empty and replaces the whole selector/page model with
  // tenant-scoped database rows. Fixtures exist only for standalone mode.
  const [orders, setOrders] = useState<ManufacturingOrder[]>(usingLiveApi ? [] : SCENARIOS);
  const [reference, setReference] = useState(usingLiveApi ? "" : SCENARIOS[0].reference);
  const [view, setView] = useState<View>("mo");
  const [loadingOrders, setLoadingOrders] = useState(usingLiveApi);
  const [loadError, setLoadError] = useState<string | null>(null);
  const order = orders.find((o) => o.reference === reference) ?? orders[0] ?? null;

  useEffect(() => {
    if (!usingLiveApi) return;
    let cancelled = false;
    setLoadingOrders(true);
    setLoadError(null);
    fetchManufacturingOrders()
      .then((result) => {
        if (cancelled) return;
        setOrders(result);
        setReference((current) =>
          result.some((candidate) => candidate.reference === current)
            ? current
            : result[0]?.reference ?? "",
        );
      })
      .catch((error) => {
        if (!cancelled) setLoadError(error instanceof Error ? error.message : String(error));
      })
      .finally(() => {
        if (!cancelled) setLoadingOrders(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleCreated = (newOrder: ManufacturingOrder) => {
    setOrders((prev) => [newOrder, ...prev.filter((row) => row.reference !== newOrder.reference)]);
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
            order ? (
              <>
                <MoHeader order={order} />
                <OrderForm key={order.reference} order={order} />
              </>
            ) : (
              <div className="card">
                <h2>{loadingOrders ? "Loading manufacturing orders…" : "No manufacturing orders"}</h2>
                {loadError && <p className="create-mo__error">{loadError}</p>}
                {!loadingOrders && !loadError && (
                  <p className="muted">No active manufacturing orders were found for this tenant.</p>
                )}
              </div>
            )
          )}
        </div>
        {view === "mo" && order && (
          <aside className="layout__side">
            <DelayRiskCard insight={order.insight} />
            <MaterialOverrunCard components={order.insight?.material_overrun ?? []} />
            <ActivityTimeline entries={order.activity} />
          </aside>
        )}
      </main>
    </div>
  );
}
