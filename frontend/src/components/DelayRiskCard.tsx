import type { ValidatedInsight } from "../types";
import { percentOfThreshold, riskBand, riskBandClass } from "../riskBand";

export function DelayRiskCard({ insight }: { insight: ValidatedInsight }) {
  const hours = insight.overrun_hours ?? 0;
  const percent = percentOfThreshold(insight.risk_score, insight.delay_threshold);
  const band = riskBand(insight.is_delayed, percent);
  const isSuppressed = insight.status === "suppressed_not_scorable" || insight.status === "rejected";

  return (
    <div className="card risk-card">
      <div className="risk-card__top">
        <div className="risk-card__icon">
          <ClockIcon />
        </div>
        <div className="risk-card__headline">
          <div className={"risk-card__hours" + (insight.is_delayed ? "" : " risk-card__hours--ok")}>
            {insight.is_delayed
              ? hours > 0
                ? `+${hours.toFixed(1)} hrs`
                : "At Risk"
              : "On Track"}
          </div>
          <div className="risk-card__hours-label">Total Estimated Delay</div>
        </div>
        <div className="risk-card__badge">
          <span className={"risk-card__percent" + (insight.is_delayed ? "" : " risk-card__percent--ok")}>
            {percent}%
          </span>
          <span className="risk-card__percent-caption">of delay threshold</span>
          <span className={riskBandClass(band)}>
            <WarnIcon /> {band}
          </span>
        </div>
      </div>

      {!isSuppressed && (
        <div className="risk-card__why">
          <h3>Why the delay is happening?</h3>
          {insight.why_lines.length === 0 ? (
            <p className="muted">No fired risk signal cleared this tenant's threshold — nothing to explain.</p>
          ) : (
            <ul>
              {insight.why_lines.map((line) => (
                <li key={line.index}>
                  <div className="risk-card__why-row">
                    <span className="risk-card__dot" />
                    <span className="risk-card__why-headline">{line.headline}</span>
                    {line.delta && <span className="risk-card__delta">{line.delta}</span>}
                  </div>
                  <div className="risk-card__why-detail">{line.detail}</div>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

function ClockIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.8" />
      <path d="M12 7v5l3.5 2" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function WarnIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
      <path
        d="M12 4l9 16H3z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
      <path d="M12 10v4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      <circle cx="12" cy="17" r="0.9" fill="currentColor" />
    </svg>
  );
}
