import type { ComponentEvidence } from "../types";

export function MaterialOverrunCard({ components }: { components: ComponentEvidence[] }) {
  return (
    <div className="card">
      <h3>Material Overrun Risk</h3>
      <p className="muted muted--tight">
        Available qty may be insufficient based on current consumption rate.
      </p>
      {components.length === 0 ? (
        <p className="muted">No short components for this order.</p>
      ) : (
        <ul className="material-list">
          {components.map((component) => (
            <li key={component.component_id}>
              <span className="material-list__dot" />
              <span className="material-list__name">{component.name ?? component.component_id}</span>
              <span className="material-list__qty">
                {formatQty(component.shortfall_quantity)} Qty
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function formatQty(value: number | null): string {
  if (value === null) return "—";
  return String(Math.round(value)).padStart(2, "0");
}
