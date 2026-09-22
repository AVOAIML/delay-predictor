import { useState, type ReactNode } from "react";
import type { ManufacturingOrder } from "../types";

export function OrderForm({ order }: { order: ManufacturingOrder }) {
  const [tab, setTab] = useState<"components" | "work-orders">("components");

  return (
    <div className="card order-form">
      <div className="form-grid">
        <Field label="Product" required>
          <select defaultValue={order.product}>
            <option>{order.product}</option>
          </select>
        </Field>
        <Field label="Quantity" required>
          <input type="text" defaultValue={order.quantity.toFixed(2)} />
        </Field>
        <Field label="Bills of Material" required>
          <select defaultValue={order.bom} disabled>
            <option>{order.bom}</option>
          </select>
        </Field>
        <Field label="Scheduled Date">
          <div className="date-input">
            <input type="text" defaultValue={order.scheduledDate} readOnly />
            <CalendarIcon />
          </div>
        </Field>
      </div>

      <div className="tabs">
        <button
          className={"tabs__tab" + (tab === "components" ? " tabs__tab--active" : "")}
          onClick={() => setTab("components")}
        >
          Components
        </button>
        <button
          className={"tabs__tab" + (tab === "work-orders" ? " tabs__tab--active" : "")}
          onClick={() => setTab("work-orders")}
        >
          Work Orders
        </button>
      </div>

      {tab === "components" ? (
        <table className="table">
          <thead>
            <tr>
              <th>Product</th>
              <th>Availability</th>
              <th className="table__num">To Consume</th>
            </tr>
          </thead>
          <tbody>
            {order.components.map((row) => (
              <tr key={row.product}>
                <td>{row.product}</td>
                <td>
                  <span className={"pill " + availabilityClass(row.availability)}>{row.availability}</span>
                </td>
                <td className="table__num">{String(row.toConsume).padStart(2, "0")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="muted">No work orders generated yet — plan this order to schedule them.</p>
      )}
    </div>
  );
}

function availabilityClass(availability: string): string {
  if (availability === "Available") return "pill--green";
  if (availability === "Short") return "pill--amber";
  return "pill--red";
}

function Field({ label, required, children }: { label: string; required?: boolean; children: ReactNode }) {
  return (
    <label className="field">
      <span className="field__label">
        {label}
        {required && <span className="field__required"> *</span>}
      </span>
      {children}
    </label>
  );
}

function CalendarIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
      <rect x="3" y="5" width="18" height="16" rx="2" stroke="currentColor" strokeWidth="1.6" />
      <path d="M3 10h18M8 3v4M16 3v4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}
