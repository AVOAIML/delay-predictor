// `risk_score` is an unbounded weighted mean of raw ratios (see types.ts) —
// there is no fixed 0..1 scale to band against. What IS meaningful is how it
// compares to `delay_threshold`, the exact comparison `is_delayed` itself is
// built from (modules/m3_production_delay/review/pipeline.py). Banding and
// the "% of threshold" framing below are this frontend's own presentation
// choice on top of that comparison, not something the module returns.
export type RiskBand = "Low Risk" | "Medium Risk" | "High Risk";

export function percentOfThreshold(riskScore: number | null, threshold: number): number {
  if (riskScore === null || threshold <= 0) return 0;
  return Math.round((riskScore / threshold) * 100);
}

export function riskBand(isDelayed: boolean | null, percentOfThresholdValue: number): RiskBand {
  if (!isDelayed) return "Low Risk";
  return percentOfThresholdValue >= 150 ? "High Risk" : "Medium Risk";
}

export function riskBandClass(band: RiskBand): string {
  switch (band) {
    case "High Risk":
      return "badge badge--high";
    case "Medium Risk":
      return "badge badge--medium";
    case "Low Risk":
      return "badge badge--low";
  }
}
