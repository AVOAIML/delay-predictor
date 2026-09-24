import type { ManufacturingOrder } from "../types";
import { StatusStepper } from "./StatusStepper";

export function MoHeader({ order }: { order: ManufacturingOrder }) {
  return (
    <div className="mo-header">
      <div className="mo-header__row">
        <div className="mo-header__title">
          <button className="icon-btn" aria-label="Back">
            <BackIcon />
          </button>
          <div>
            <h1>{order.reference}</h1>
            <p className="mo-header__subtitle">{order.product}</p>
          </div>
        </div>
        <div className="mo-header__actions">
          <button className="btn btn--outline">Cancel MO</button>
          <button className="btn btn--outline">
            <PrintIcon /> Print
          </button>
          <button className="btn btn--primary">
            <PlanIcon /> Plan
          </button>
        </div>
      </div>
      <StatusStepper stage={order.stage} />
    </div>
  );
}

function BackIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
      <path d="M15 19l-7-7 7-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function PrintIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
      <path d="M6 9V3h12v6" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
      <rect x="4" y="9" width="16" height="8" rx="1.2" stroke="currentColor" strokeWidth="1.8" />
      <rect x="7" y="14" width="10" height="7" stroke="currentColor" strokeWidth="1.8" />
    </svg>
  );
}

function PlanIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
      <rect x="4" y="4" width="16" height="16" rx="2" stroke="currentColor" strokeWidth="1.8" />
      <path d="M9 12h6M12 9v6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}
