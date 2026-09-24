import { useEffect, useMemo, useState, type FormEvent } from "react";
import { createManufacturingOrder, fetchManufacturingOrderOptions } from "../api";
import type { BomOption, CreateMoComponentInput, CreateWorkOrderInput, ManufacturingOrder, ProductOption } from "../types";

interface Props { onCreated: (order: ManufacturingOrder) => void; onCancel: () => void }

function fromBom(bom: BomOption): CreateWorkOrderInput[] {
  const positions = new Map(bom.operations.map((op, index) => [op.operation_id, index]));
  return bom.operations.map((op) => ({
    operation_id: op.operation_id, name: op.name,
    work_center_id: op.work_center_id, work_center_name: op.work_center_name,
    expected_duration_hours: op.expected_duration_hours,
    actual_duration_hours: null, units_done: 0,
    depends_on_index: op.depends_on_operation_id ? positions.get(op.depends_on_operation_id) ?? null : null,
  }));
}

export function CreateMoPanel({ onCreated, onCancel }: Props) {
  const [products, setProducts] = useState<ProductOption[]>([]);
  const [productId, setProductId] = useState("");
  const [bomId, setBomId] = useState("");
  const [quantity, setQuantity] = useState(1);
  const [threshold, setThreshold] = useState(1);
  const [components, setComponents] = useState<CreateMoComponentInput[]>([]);
  const [workOrders, setWorkOrders] = useState<CreateWorkOrderInput[]>([]);
  const [detailTab, setDetailTab] = useState<"components" | "work-orders">("work-orders");
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const product = useMemo(() => products.find((p) => p.id === productId) ?? null, [products, productId]);
  const bom = useMemo(() => product?.boms.find((b) => b.id === bomId) ?? null, [product, bomId]);

  const chooseBom = (next: BomOption | null) => {
    setBomId(next?.id ?? "");
    setComponents((next?.components ?? []).map(({ name, required_quantity, available_quantity }) => ({
      name,
      required_quantity: required_quantity * quantity,
      available_quantity,
    })));
    setWorkOrders(next ? fromBom(next) : []);
  };

  const changeQuantity = (next: number) => {
    if (next > 0 && quantity > 0) {
      const scale = next / quantity;
      setComponents((rows) => rows.map((row) => ({
        ...row,
        required_quantity: row.required_quantity * scale,
      })));
    }
    setQuantity(next);
  };

  useEffect(() => {
    fetchManufacturingOrderOptions().then((rows) => {
      setProducts(rows);
      if (rows[0]) setProductId(rows[0].id);
      chooseBom(rows[0]?.boms[0] ?? null);
    }).catch((reason) => setError(String(reason))).finally(() => setLoading(false));
  }, []);

  const updateComponent = (i: number, patch: Partial<CreateMoComponentInput>) =>
    setComponents((rows) => rows.map((row, index) => index === i ? { ...row, ...patch } : row));
  const updateWorkOrder = (i: number, patch: Partial<CreateWorkOrderInput>) =>
    setWorkOrders((rows) => rows.map((row, index) => index === i ? { ...row, ...patch } : row));

  const removeWorkOrder = (removed: number) => setWorkOrders((rows) =>
    rows.filter((_, i) => i !== removed).map((row) => ({
      ...row,
      depends_on_index: row.depends_on_index === removed ? null
        : row.depends_on_index !== null && row.depends_on_index > removed ? row.depends_on_index - 1
        : row.depends_on_index,
    })),
  );

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setSubmitting(true); setError(null);
    try {
      const response = await createManufacturingOrder({
        product_id: productId, bom_id: bomId, quantity, threshold,
        components: components.filter((row) => row.name.trim()), work_orders: workOrders,
      });
      onCreated(response.order);
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setSubmitting(false); }
  };

  if (loading) return <div className="card"><p className="muted">Loading Product and BOM data…</p></div>;
  if (!products.length) return <div className="card"><p className="create-mo__error">No active Products found.</p></div>;

  return <div className="card create-mo">
    <div className="create-mo__header"><h2>Create and score a Manufacturing Order</h2>
      <p className="muted">Product, BOM, components and operations load from the tenant database. Adjust the MO snapshot before scoring.</p>
    </div>
    <form onSubmit={submit}>
      <div className="form-grid">
        <label className="field"><span className="field__label">Product</span>
          <select value={productId} onChange={(e) => { const p = products.find((row) => row.id === e.target.value); setProductId(e.target.value); chooseBom(p?.boms[0] ?? null); }}>
            {products.map((p) => <option key={p.id} value={p.id}>{p.name} · {p.sku}</option>)}
          </select></label>
        <label className="field"><span className="field__label">Quantity</span>
          <input type="number" min="0.01" step="any" value={quantity} onChange={(e) => changeQuantity(Number(e.target.value))} /></label>
        <label className="field"><span className="field__label">Bill of Material</span>
          <select value={bomId} onChange={(e) => chooseBom(product?.boms.find((row) => row.id === e.target.value) ?? null)}>
            {(product?.boms ?? []).map((b) => <option key={b.id} value={b.id}>{b.name} · {b.code}</option>)}
          </select></label>
        <label className="field"><span className="field__label">Delay threshold</span>
          <input type="number" min="0.01" step="any" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} /></label>
      </div>

      <div className="tabs create-mo__tabs">
        <button type="button" className={"tabs__tab" + (detailTab === "components" ? " tabs__tab--active" : "")} onClick={() => setDetailTab("components")}>Components</button>
        <button type="button" className={"tabs__tab" + (detailTab === "work-orders" ? " tabs__tab--active" : "")} onClick={() => setDetailTab("work-orders")}>Work Orders</button>
      </div>

      {detailTab === "components" ? <div className="create-mo__components">
        <div className="create-mo__component-row create-mo__component-row--header"><span>Name</span><span>Required</span><span>Available</span><span /></div>
        {components.map((row, i) => <div className="create-mo__component-row" key={i}>
          <input value={row.name} onChange={(e) => updateComponent(i, { name: e.target.value })} />
          <input type="number" min="0" step="any" value={row.required_quantity} onChange={(e) => updateComponent(i, { required_quantity: Number(e.target.value) })} />
          <input type="number" min="0" step="any" value={row.available_quantity} onChange={(e) => updateComponent(i, { available_quantity: Number(e.target.value) })} />
          <button type="button" className="icon-btn" aria-label="Remove component" onClick={() => setComponents((rows) => rows.filter((_, index) => index !== i))}>×</button>
        </div>)}
        <button type="button" className="btn btn--outline btn--small create-mo__add" onClick={() => setComponents((rows) => [...rows, { name: "", required_quantity: 1, available_quantity: 1 }])}>+ Add component</button>
      </div> : <div className="create-mo__work-orders">
        <div className="create-mo__table-wrap"><table className="table work-orders-table">
          <thead><tr><th>Operation</th><th>Work Center</th><th>Quantity</th><th>Estimated Duration</th><th>Actual Duration</th><th aria-label="Start" /><th>Depends On</th><th>Status</th><th /></tr></thead>
          <tbody>{workOrders.map((row, i) => <tr key={`${row.operation_id ?? "new"}-${i}`}>
            <td><input aria-label={`Operation ${i + 1}`} value={row.name} readOnly={row.operation_id !== null} onChange={(e) => updateWorkOrder(i, { name: e.target.value })} /></td>
            <td><input aria-label={`Work center ${i + 1}`} value={row.work_center_name} readOnly={row.work_center_id !== null} onChange={(e) => updateWorkOrder(i, { work_center_name: e.target.value })} /></td>
            <td><input aria-label={`Units completed ${i + 1}`} type="number" min="0" max={quantity} step="any" value={row.units_done} onChange={(e) => updateWorkOrder(i, { units_done: Number(e.target.value) })} /><span className="work-orders-table__total">/ {quantity}</span></td>
            <td><div className="duration-input"><input aria-label={`Estimated duration ${i + 1}`} type="number" min="0.01" step="any" value={row.expected_duration_hours} onChange={(e) => updateWorkOrder(i, { expected_duration_hours: Number(e.target.value) })} /><span>hrs</span></div></td>
            <td><div className="duration-input"><input aria-label={`Actual duration ${i + 1}`} type="number" min="0" step="any" value={row.actual_duration_hours ?? ""} placeholder="00:00" onChange={(e) => updateWorkOrder(i, { actual_duration_hours: e.target.value === "" ? null : Number(e.target.value) })} /><span>hrs</span></div></td>
            <td><button type="button" className="work-order-start" aria-label={`Start ${row.name}`} onClick={() => updateWorkOrder(i, { actual_duration_hours: row.actual_duration_hours && row.actual_duration_hours > 0 ? row.actual_duration_hours : 0.02 })}><span aria-hidden="true">▶</span></button></td>
            <td><select aria-label={`Dependency ${i + 1}`} value={row.depends_on_index ?? ""} disabled={row.operation_id !== null} onChange={(e) => updateWorkOrder(i, { depends_on_index: e.target.value === "" ? null : Number(e.target.value) })}><option value="">Independent</option>{workOrders.slice(0, i).map((prior, p) => <option key={p} value={p}>{prior.name}</option>)}</select></td>
            <td><span className={(row.actual_duration_hours ?? 0) > 0 ? "work-order-status work-order-status--progress" : "work-order-status"}>{(row.actual_duration_hours ?? 0) > 0 ? "In Progress" : "To Do"}</span></td>
            <td><button type="button" className="icon-btn" aria-label="Remove work order" onClick={() => removeWorkOrder(i)}>×</button></td>
          </tr>)}</tbody>
        </table></div>
        <button type="button" className="btn btn--outline btn--small create-mo__add" onClick={() => setWorkOrders((rows) => [...rows, { operation_id: null, work_center_id: null, work_center_name: "Frontend MO Work Center", name: "New Operation", expected_duration_hours: 1, actual_duration_hours: null, units_done: 0, depends_on_index: rows.length ? rows.length - 1 : null }])}>+ Add work order</button>
      </div>}

      {error && <p className="create-mo__error">{error}</p>}
      <div className="create-mo__actions"><button type="button" className="btn btn--outline" onClick={onCancel}>Cancel</button>
        <button type="submit" className="btn btn--primary" disabled={submitting || !bom || !workOrders.length}>{submitting ? "Creating & scoring…" : "Create & Score"}</button></div>
    </form>
  </div>;
}
