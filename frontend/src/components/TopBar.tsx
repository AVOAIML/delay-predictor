import type { ManufacturingOrder } from "../types";

interface Props {
  orders: ManufacturingOrder[];
  selected: string;
  onSelect: (reference: string) => void;
  onCreateClick?: () => void;
  onWeightsClick?: () => void;
}

export function TopBar({ orders, selected, onSelect, onCreateClick, onWeightsClick }: Props) {
  return (
    <header className="topbar">
      <div className="topbar__left">
        <button className="icon-btn" aria-label="Apps">
          <GridIcon />
        </button>
        <nav className="breadcrumb" aria-label="Breadcrumb">
          <span>Home</span>
          <ChevronIcon />
          <span>Manufacturing</span>
          <ChevronIcon />
          <span>Manufacturing Orders</span>
          <ChevronIcon />
          <span className="breadcrumb__active">Create Manufacturing Order</span>
        </nav>
      </div>
      <div className="topbar__right">
        <label className="demo-switcher">
          <span>Demo MO</span>
          <select value={selected} onChange={(e) => onSelect(e.target.value)}>
            {orders.map((order) => (
              <option key={order.reference} value={order.reference}>
                {order.reference} · {order.product}
              </option>
            ))}
          </select>
        </label>
        {onCreateClick && (
          <button className="btn btn--outline btn--small" onClick={onCreateClick}>
            + New MO
          </button>
        )}
        {onWeightsClick && (
          <button className="btn btn--outline btn--small" onClick={onWeightsClick}>
            Weights (Admin)
          </button>
        )}
        <button className="icon-btn" aria-label="Notifications">
          <BellIcon />
        </button>
        <div className="avatar avatar--purple">SS</div>
        <span className="user-name">Sam Smith</span>
      </div>
    </header>
  );
}

function GridIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <rect x="3" y="3" width="7" height="7" rx="1.5" fill="currentColor" />
      <rect x="14" y="3" width="7" height="7" rx="1.5" fill="currentColor" />
      <rect x="3" y="14" width="7" height="7" rx="1.5" fill="currentColor" />
      <rect x="14" y="14" width="7" height="7" rx="1.5" fill="currentColor" />
    </svg>
  );
}

function ChevronIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
      <path d="M9 6l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function BellIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <path
        d="M12 3a5 5 0 00-5 5v3.5c0 .8-.3 1.5-.9 2.1L5 15h14l-1.1-1.4a3 3 0 01-.9-2.1V8a5 5 0 00-5-5z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <path d="M10 18a2 2 0 004 0" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}
