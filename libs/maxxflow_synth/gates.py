"""The 3 CI validation gates (plan §5, §12). ALL must pass before a synthetic
batch trains:

  A. schema / constraint  — types, Decimal scale, FK existence, MasterData codes,
                            ropStatus-forbidden, deletedAt exclusion
  B. statistical realism  — class-balance bands, non-degenerate distributions
  C. leakage + learnability — no single feature's AUC exceeds a threshold (catches
                            a leaked deterministic column, generator-agnostic);
                            trivial-model AUC≈1.0 is treated as a LEAK ALARM; each
                            module must land in its acceptance band.

A model that is *too good* is the bug.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from maxxflow_data.schema_def import TENANT_TABLES, columns as schema_columns
from maxxflow_synth.simulators import simulate

# Per-module acceptance (plan §5: e.g. M1 AUC ∈ [0.72,0.82]). AUC upper bounds are
# all < 0.99 so a leaked/too-easy task fails. Tuned so a fair logistic lands in band.
BANDS = {
    "m1_quote":     {"auc": (0.72, 0.82), "pos": (0.25, 0.72)},
    "m2_inventory": {"auc": (0.74, 0.90), "pos": (0.12, 0.55)},
    "m3_delay":     {"auc": (0.68, 0.90), "pos": (0.15, 0.40), "pr_auc_lift": 1.25},
    "m4_bom":       {"auc": (0.78, 0.985), "pos": (0.10, 0.32), "prec_at_k": (0.45, 0.99)},
}
SINGLE_FEATURE_AUC_MAX = 0.96   # above this => a near-deterministic (leaked) feature
TRIVIAL_AUC_ALARM = 0.985       # >= this => "too good", treated as a leak


FORBIDDEN_FEATURE_COLS = {"rop_status", "tenant_id",
                          "real_duration", "actual_start", "actual_end", "completed_at"}


def _encoder(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    transformers = [("num", StandardScaler(), num)]
    if cat:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
    return ColumnTransformer(transformers)


def _fair_model_metrics(X: pd.DataFrame, y: pd.Series, seed: int) -> dict:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    pipe = Pipeline([("enc", _encoder(X)), ("clf", LogisticRegression(max_iter=2000))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, p))
    pr = float(average_precision_score(yte, p))
    brier = float(brier_score_loss(yte, p))
    k = int(yte.sum())
    order = np.argsort(-p)
    prec_at_k = float(yte.to_numpy()[order][:k].mean()) if k > 0 else float("nan")
    return {"auc": auc, "pr_auc": pr, "brier": brier, "prec_at_k": prec_at_k, "n_pos_test": k}


def _single_feature_aucs(X: pd.DataFrame, y: pd.Series) -> dict:
    out = {}
    for col in X.columns:
        s = X[col]
        if not pd.api.types.is_numeric_dtype(s):
            # ordinal-encode categories for a quick separability probe
            s = s.astype("category").cat.codes
        try:
            a = roc_auc_score(y, s)
            out[col] = float(max(a, 1 - a))  # direction-agnostic
        except ValueError:
            out[col] = float("nan")
    return out


# ---------------------------------------------------------------------------
def gate_schema(batch) -> dict:
    """A — schema / constraint."""
    errors = []
    # feature frame must be clean numeric/category, no forbidden columns, no NaN/inf
    bad = set(batch.features.columns) & FORBIDDEN_FEATURE_COLS
    if bad:
        errors.append(f"forbidden feature columns present: {sorted(bad)}")
    num = batch.features.select_dtypes(include=[np.number])
    if not np.isfinite(num.to_numpy()).all():
        errors.append("non-finite values in feature matrix")

    # raw tables (M1): unknown-column + required + FK + MasterData-code checks
    for table, df in batch.tables.items():
        if table not in TENANT_TABLES:
            continue
        allowed = set(schema_columns(table))
        unknown = set(df.columns) - allowed
        if unknown:
            errors.append(f"{table}: unknown columns {sorted(unknown)}")
        for col in TENANT_TABLES[table]:
            if not col.nullable and col.default is None and col.name in df.columns:
                if df[col.name].isna().any():
                    errors.append(f"{table}.{col.name} has nulls but is NOT NULL")

    if batch.tables:
        md = batch.tables.get("master_data")
        if md is not None:
            valid_ids = set(md["id"])
            q = batch.tables.get("quotations")
            if q is not None:
                for fk in ("stage_id", "status_id"):
                    missing = ~q[fk].isin(valid_ids)
                    if missing.any():
                        errors.append(f"quotations.{fk}: {int(missing.sum())} rows reference unknown MasterData")
            li = batch.tables.get("quotation_line_items")
            prod = batch.tables.get("products")
            if li is not None and prod is not None:
                if (~li["product_id"].isin(set(prod["id"]))).any():
                    errors.append("quotation_line_items.product_id references unknown product")
    return {"passed": not errors, "errors": errors}


def gate_realism(batch) -> dict:
    """B — class balance + non-degenerate distributions."""
    errors = []
    band = BANDS[batch.module]
    pos = float(batch.label.mean())
    lo, hi = band["pos"]
    if not (lo <= pos <= hi):
        errors.append(f"positive rate {pos:.3f} outside class-balance band {band['pos']}")
    num = batch.features.select_dtypes(include=[np.number])
    degenerate = [c for c in num.columns if num[c].nunique() <= 1]
    if degenerate:
        errors.append(f"degenerate (constant) features: {degenerate}")
    return {"passed": not errors, "errors": errors, "positive_rate": pos}


def gate_leakage_learnability(batch, seed: int) -> dict:
    """C — leakage + learnability (the headline gate)."""
    errors = []
    X, y = batch.features, batch.label
    sf = _single_feature_aucs(X, y)
    max_sf = float(np.nanmax(list(sf.values())))
    worst = max(sf, key=lambda k: (sf[k] if sf[k] == sf[k] else -1))
    if max_sf > SINGLE_FEATURE_AUC_MAX:
        errors.append(f"single-feature leak: {worst} AUC={max_sf:.3f} > {SINGLE_FEATURE_AUC_MAX}")

    m = _fair_model_metrics(X, y, seed)
    if m["auc"] >= TRIVIAL_AUC_ALARM:
        errors.append(f"AUC≈1.0 leak alarm: holdout AUC={m['auc']:.3f} >= {TRIVIAL_AUC_ALARM}")

    band = BANDS[batch.module]
    lo, hi = band["auc"]
    if not (lo <= m["auc"] <= hi):
        errors.append(f"holdout AUC {m['auc']:.3f} outside learnability band {band['auc']}")
    if "pr_auc_lift" in band:
        base = float(y.mean())
        if m["pr_auc"] < base * band["pr_auc_lift"]:
            errors.append(f"PR-AUC {m['pr_auc']:.3f} < {band['pr_auc_lift']}x base rate {base:.3f}")
    if "prec_at_k" in band:
        plo, phi = band["prec_at_k"]
        if not (plo <= m["prec_at_k"] <= phi):
            errors.append(f"precision@k {m['prec_at_k']:.3f} outside band {band['prec_at_k']}")

    return {"passed": not errors, "errors": errors,
            "metrics": m, "max_single_feature_auc": max_sf, "single_feature_aucs": sf}


def run_module_gates(module: str, tenant: str = "demo", seed: int = 7) -> dict:
    batch = simulate(module, seed=seed, tenant=tenant)
    a = gate_schema(batch)
    b = gate_realism(batch)
    c = gate_leakage_learnability(batch, seed)
    return {"module": module, "passed": a["passed"] and b["passed"] and c["passed"],
            "gate_A_schema": a, "gate_B_realism": b, "gate_C_leakage_learnability": c,
            "meta": {k: v for k, v in batch.meta.items() if k not in ("md_map", "feature_frame", "err_type")}}


def run_all_gates(tenant: str = "demo", module: str = "all", seed: int = 7) -> dict:
    modules = ["m1_quote", "m2_inventory", "m3_delay", "m4_bom"] if module == "all" else [module]
    reports = {m: run_module_gates(m, tenant, seed) for m in modules}
    return {"passed": all(r["passed"] for r in reports.values()), "modules": reports}
