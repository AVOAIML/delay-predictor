import type { MoStage } from "../types";

const STEPS: { key: MoStage; label: string }[] = [
  { key: "draft", label: "Draft" },
  { key: "confirmed", label: "Confirmed" },
  { key: "done", label: "Done" },
];

export function StatusStepper({ stage }: { stage: MoStage }) {
  const activeIndex = STEPS.findIndex((s) => s.key === stage);

  return (
    <div className="stepper">
      <div className="stepper__track">
        {STEPS.map((step, i) => (
          <div className="stepper__step" key={step.key}>
            <div
              className={
                "stepper__node" +
                (i < activeIndex ? " stepper__node--done" : "") +
                (i === activeIndex ? " stepper__node--active" : "")
              }
            >
              <StageIcon stage={step.key} />
            </div>
            <span className={"stepper__label" + (i <= activeIndex ? " stepper__label--active" : "")}>
              {step.label}
            </span>
            {i < STEPS.length - 1 && (
              <div className={"stepper__connector" + (i < activeIndex ? " stepper__connector--done" : "")} />
            )}
          </div>
        ))}
      </div>
      <button className="icon-btn icon-btn--outline" aria-label="History">
        <HistoryIcon />
      </button>
    </div>
  );
}

function StageIcon({ stage }: { stage: MoStage }) {
  if (stage === "done") {
    return (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
        <path d="M4 12l5 5L20 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  }
  if (stage === "confirmed") {
    return (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
        <rect x="4" y="7" width="16" height="13" rx="1.5" stroke="currentColor" strokeWidth="1.8" />
        <path d="M4 11h16" stroke="currentColor" strokeWidth="1.8" />
        <path d="M9 4v5M15 4v5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      </svg>
    );
  }
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
      <path d="M6 3h9l5 5v13H6z" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
      <path d="M15 3v5h5" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
    </svg>
  );
}

function HistoryIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <path
        d="M3 12a9 9 0 109-9 9 9 0 00-6.4 2.7L3 8"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M3 4v4h4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M12 7v5l3 3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
