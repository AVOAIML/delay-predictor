import { useState } from "react";
import { resolveWeights } from "../api";
import {
  WEIGHT_BOUNDS_BP,
  WEIGHT_PRIOR_BP,
  WEIGHT_SIGNAL_ORDER,
  type ManufacturingOrder,
  type ResolveWeightsResponse,
  type WeightSignalName,
} from "../types";
import { DelayRiskCard } from "./DelayRiskCard";
import { MaterialOverrunCard } from "./MaterialOverrunCard";

const SIGNAL_LABELS: Record<WeightSignalName, string> = {
  time_overrun: "Time overrun",
  operator_skill: "Operator skill",
  seasonality: "Seasonality",
  material_availability: "Material availability",
  supplier_reliability: "Supplier reliability",
};

interface Props {
  orders: ManufacturingOrder[];
  defaultJobReference: string;
  onRescored: (jobReference: string, insight: ResolveWeightsResponse["insight"]) => void;
  onClose: () => void;
}

type Mode = "add" | "skip";

export function WeightsAdminPanel({ orders, defaultJobReference, onRescored, onClose }: Props) {
  const [step, setStep] = useState<"configure" | "dashboard">("configure");
  const [mode, setMode] = useState<Mode>("add");
  const [jobReference, setJobReference] = useState(defaultJobReference);
  const [threshold, setThreshold] = useState(1.0);
  const [weightsBp, setWeightsBp] = useState<Record<WeightSignalName, number>>({ ...WEIGHT_PRIOR_BP });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ResolveWeightsResponse | null>(null);

  const sum = WEIGHT_SIGNAL_ORDER.reduce((total, signal) => total + weightsBp[signal], 0);
  const sumValid = sum === 10000;
  const canSubmit = mode === "skip" || sumValid;

  const handleSubmit = async () => {
    setError(null);
    setSubmitting(true);
    try {
      const response = await resolveWeights({
        configured_bp: mode === "add" ? weightsBp : null,
        job_reference: jobReference,
        threshold,
      });
      setResult(response);
      onRescored(jobReference, response.insight);
      setStep("dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  if (step === "dashboard" && result) {
    const resolution = result.weight_resolution;
    const rejected = resolution.fallback_reasons.some((r) => r.startsWith("configured_weights_invalid"));
    return (
      <div className="card weights-dashboard">
        <div className="weights-dashboard__header">
          <h2>Weight Agent — result</h2>
          <button className="btn btn--outline btn--small" onClick={() => setStep("configure")}>
            ← Configure again
          </button>
        </div>

        {rejected && (
          <p className="create-mo__error">
            Your configured weights were rejected and the agent fell back to the prior — see
            "Fallback reasons" below.
          </p>
        )}

        <div className="weights-dashboard__grid">
          <div>
            <div className="weights-dashboard__meta">
              <span className={`badge ${resolution.source === "configured" ? "badge--low" : "badge--medium"}`}>
                source: {resolution.source}
              </span>
              <span className="badge badge--medium">status: {resolution.status}</span>
              <span className="muted">confidence {(resolution.confidence * 100).toFixed(0)}%</span>
            </div>

            <h3>Resolved weights</h3>
            <ul className="weights-bars">
              {WEIGHT_SIGNAL_ORDER.map((signal) => (
                <li key={signal}>
                  <span className="weights-bars__label">{SIGNAL_LABELS[signal]}</span>
                  <div className="weights-bars__track">
                    <div
                      className="weights-bars__fill"
                      style={{ width: `${resolution.weights_bp[signal] / 100}%` }}
                    />
                  </div>
                  <span className="weights-bars__value">{(resolution.weights_bp[signal] / 100).toFixed(0)}%</span>
                </li>
              ))}
            </ul>

            {resolution.fallback_reasons.length > 0 && (
              <>
                <h3>Fallback reasons</h3>
                <ul className="weights-reasons">
                  {resolution.fallback_reasons.map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                </ul>
              </>
            )}
          </div>

          <div>
            <h3>Re-scored: {result.job_id}</h3>
            <p className="muted muted--tight">
              Same job, scored with the weights above instead of whatever it was scored with before.
            </p>
            <DelayRiskCard insight={result.insight} />
          </div>
        </div>

        <MaterialOverrunCard components={result.insight.material_overrun} />

        <div className="create-mo__actions">
          <button className="btn btn--primary" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="card create-mo">
      <div className="create-mo__header">
        <h2>Admin: Configure Weights</h2>
        <p className="muted">
          Calls the real Weight Agent (<code>ProductionDelayOrchestrator.resolve_weights</code>) and
          re-scores a job with exactly what it resolves — proving whether configuring weights (or
          skipping to let it cold-start) actually changes the outcome.
        </p>
      </div>

      <div className="weights-mode">
        <label>
          <input type="radio" checked={mode === "add"} onChange={() => setMode("add")} />
          Add — configure explicit weights
        </label>
        <label>
          <input type="radio" checked={mode === "skip"} onChange={() => setMode("skip")} />
          Skip — let the agent cold-start (falls back to the prior locally, since the stub LLM
          returns no usable profile)
        </label>
      </div>

      <div className="form-grid">
        <label className="field">
          <span className="field__label">Target MO</span>
          <select value={jobReference} onChange={(e) => setJobReference(e.target.value)}>
            {orders.map((o) => (
              <option key={o.reference} value={o.reference}>
                {o.reference} · {o.product}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span className="field__label">Delay threshold</span>
          <input
            type="number" min="0" step="any" value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
          />
        </label>
      </div>

      {mode === "add" && (
        <div className="weights-inputs">
          <div className={"weights-sum" + (sumValid ? " weights-sum--ok" : " weights-sum--bad")}>
            Sum: {sum} / 10000 {sumValid ? "✓" : "— must equal exactly 10000"}
          </div>
          {WEIGHT_SIGNAL_ORDER.map((signal) => {
            const bounds = WEIGHT_BOUNDS_BP[signal];
            return (
              <label className="field weights-inputs__row" key={signal}>
                <span className="field__label">
                  {SIGNAL_LABELS[signal]}
                  <span className="muted"> (bounds {bounds.min}–{bounds.max} bp)</span>
                </span>
                <input
                  type="number" step="any" value={weightsBp[signal]}
                  onChange={(e) =>
                    setWeightsBp((prev) => ({ ...prev, [signal]: Number(e.target.value) }))
                  }
                />
              </label>
            );
          })}
        </div>
      )}

      {error && <p className="create-mo__error">{error}</p>}

      <div className="create-mo__actions">
        <button className="btn btn--outline" onClick={onClose} disabled={submitting}>
          Cancel
        </button>
        <button className="btn btn--primary" onClick={handleSubmit} disabled={submitting || !canSubmit}>
          {submitting ? "Resolving & re-scoring…" : "Resolve & Re-score"}
        </button>
      </div>
    </div>
  );
}
