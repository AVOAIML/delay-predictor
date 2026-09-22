import { useState, type FormEvent } from "react";
import { createManufacturingOrder } from "../api";
import type { CreateMoComponentInput, ManufacturingOrder } from "../types";

interface Props {
  onCreated: (order: ManufacturingOrder) => void;
  onCancel: () => void;
}

function today(): string {
  const d = new Date();
  return `${String(d.getDate()).padStart(2, "0")}/${String(d.getMonth() + 1).padStart(2, "0")}/${d.getFullYear()}`;
}

export function CreateMoPanel({ onCreated, onCancel }: Props) {
  const [product, setProduct] = useState("Office Desk");
  const [quantity, setQuantity] = useState(1);
  const [operationName, setOperationName] = useState("Assemble Desk");
  const [expectedHours, setExpectedHours] = useState(3);
  const [actualHours, setActualHours] = useState(6);
  const [threshold, setThreshold] = useState(1.0);
  const [components, setComponents] = useState<CreateMoComponentInput[]>([
    { name: "Desk Legs", required_quantity: 8, available_quantity: 6 },
  ]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updateComponent = (index: number, patch: Partial<CreateMoComponentInput>) => {
    setComponents((rows) => rows.map((row, i) => (i === index ? { ...row, ...patch } : row)));
  };

  const addComponent = () => {
    setComponents((rows) => [...rows, { name: "", required_quantity: 1, available_quantity: 1 }]);
  };

  const removeComponent = (index: number) => {
    setComponents((rows) => rows.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const cleanComponents = components.filter((c) => c.name.trim().length > 0);
      const response = await createManufacturingOrder({
        product,
        quantity,
        operation_name: operationName,
        expected_duration_hours: expectedHours,
        actual_duration_hours: actualHours,
        threshold,
        components: cleanComponents,
      });
      const order: ManufacturingOrder = {
        reference: response.job_id,
        product,
        quantity,
        bom: `${product} BoM`,
        scheduledDate: today(),
        stage: "confirmed",
        components: cleanComponents.map((c) => ({
          product: c.name,
          availability: c.available_quantity < c.required_quantity ? "Short" : "Available",
          toConsume: c.required_quantity,
        })),
        activity: [
          {
            actor: "You",
            initials: "YO",
            timestamp: new Date().toLocaleString("en-AU", {
              day: "2-digit", month: "short", year: "numeric", hour: "numeric", minute: "2-digit",
            }),
            title: "Manufacturing Order Created",
          },
        ],
        insight: response.insight,
      };
      onCreated(order);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="card create-mo">
      <div className="create-mo__header">
        <h2>Create a test Manufacturing Order</h2>
        <p className="muted">
          Inserts real rows into <code>tenant_demo</code> and runs the actual M3 pipeline against
          them — the delay panel on the right will show whatever it genuinely computes, not a
          preset.
        </p>
      </div>
      <form onSubmit={handleSubmit}>
        <div className="form-grid">
          <label className="field">
            <span className="field__label">Product</span>
            <input value={product} onChange={(e) => setProduct(e.target.value)} required />
          </label>
          <label className="field">
            <span className="field__label">Quantity</span>
            <input
              type="number" min="0" step="any" value={quantity}
              onChange={(e) => setQuantity(Number(e.target.value))} required
            />
          </label>
          <label className="field">
            <span className="field__label">Operation name</span>
            <input value={operationName} onChange={(e) => setOperationName(e.target.value)} required />
          </label>
          <label className="field">
            <span className="field__label">Delay threshold</span>
            <input
              type="number" min="0" step="any" value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))} required
            />
          </label>
          <label className="field">
            <span className="field__label">Expected duration (hrs)</span>
            <input
              type="number" min="0" step="any" value={expectedHours}
              onChange={(e) => setExpectedHours(Number(e.target.value))} required
            />
          </label>
          <label className="field">
            <span className="field__label">Actual time logged (hrs)</span>
            <input
              type="number" min="0" step="any" value={actualHours}
              onChange={(e) => setActualHours(Number(e.target.value))} required
            />
          </label>
        </div>

        <div className="create-mo__components">
          <span className="field__label">Components</span>
          <div className="create-mo__component-row create-mo__component-row--header">
            <span>Component name</span>
            <span>Required qty</span>
            <span>Available qty</span>
            <span />
          </div>
          {components.map((row, i) => (
            <div className="create-mo__component-row" key={i}>
              <input
                aria-label="Component name" placeholder="e.g. Steel Rod" value={row.name}
                onChange={(e) => updateComponent(i, { name: e.target.value })}
              />
              <input
                aria-label="Required quantity" type="number" min="0" step="any"
                placeholder="Required qty" value={row.required_quantity}
                onChange={(e) => updateComponent(i, { required_quantity: Number(e.target.value) })}
              />
              <input
                aria-label="Available quantity" type="number" min="0" step="any"
                placeholder="Available qty" value={row.available_quantity}
                onChange={(e) => updateComponent(i, { available_quantity: Number(e.target.value) })}
              />
              <button type="button" className="icon-btn" onClick={() => removeComponent(i)} aria-label="Remove component">
                ×
              </button>
            </div>
          ))}
          <button type="button" className="btn btn--outline btn--small" onClick={addComponent}>
            + Add component
          </button>
        </div>

        {error && <p className="create-mo__error">{error}</p>}

        <div className="create-mo__actions">
          <button type="button" className="btn btn--outline" onClick={onCancel} disabled={submitting}>
            Cancel
          </button>
          <button type="submit" className="btn btn--primary" disabled={submitting}>
            {submitting ? "Creating & scoring…" : "Create & Score"}
          </button>
        </div>
      </form>
    </div>
  );
}
