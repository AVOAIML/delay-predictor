"""Drift reporting (plan §8). Evidently when installed (HTML report == prod Azure
ML monitor); otherwise a dependency-free PSI fallback so ``make drift`` always
works on a modest laptop. Same job, sink differs."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


def population_stability_index(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) < 2 or len(cur) < 2:
        return 0.0
    quantiles = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    r = np.histogram(ref, bins=quantiles)[0] / len(ref)
    c = np.histogram(cur, bins=quantiles)[0] / len(cur)
    eps = 1e-6
    r = np.clip(r, eps, None)
    c = np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


class DriftReporter:
    """Implements :class:`maxxflow_core.ports.DriftReporter`."""

    PSI_THRESHOLD = 0.2  # >0.2 = significant population shift

    def report(self, reference: pd.DataFrame, current: pd.DataFrame, *, name: str, out_dir: str) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        cols = [c for c in reference.columns if c in current.columns
                and pd.api.types.is_numeric_dtype(reference[c])]
        psi = {c: population_stability_index(reference[c].to_numpy(float), current[c].to_numpy(float))
               for c in cols}
        drift_detected = any(v > self.PSI_THRESHOLD for v in psi.values())
        result = {"name": name, "n_reference": len(reference), "n_current": len(current),
                  "psi": psi, "drift_detected": drift_detected, "engine": "psi-fallback"}

        # Prefer Evidently if available (parity with the prod monitor).
        try:  # pragma: no cover - optional heavy dep
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report
            rep = Report(metrics=[DataDriftPreset()])
            rep.run(reference_data=reference[cols], current_data=current[cols])
            html_path = os.path.join(out_dir, f"{name}_drift.html")
            rep.save_html(html_path)
            result["engine"] = "evidently"
            result["report_path"] = html_path
            return result
        except Exception:
            pass

        # Fallback HTML + JSON
        report_path = os.path.join(out_dir, f"{name}_drift.html")
        rows = "".join(
            f"<tr><td>{c}</td><td>{v:.4f}</td><td>{'DRIFT' if v > self.PSI_THRESHOLD else 'ok'}</td></tr>"
            for c, v in psi.items()
        )
        with open(report_path, "w") as fh:
            fh.write(f"<h2>Drift report: {name}</h2>"
                     f"<p>reference={len(reference)} current={len(current)} "
                     f"drift_detected={drift_detected}</p>"
                     f"<table border=1><tr><th>feature</th><th>PSI</th><th>status</th></tr>{rows}</table>")
        with open(os.path.join(out_dir, f"{name}_drift.json"), "w") as fh:
            json.dump(result, fh, indent=2)
        result["report_path"] = report_path
        return result


def get_drift_reporter() -> DriftReporter:
    return DriftReporter()
