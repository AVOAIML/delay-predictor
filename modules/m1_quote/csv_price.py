"""M1 Smart Quote Optimiser — PRICE BAND model (regression), CSV or DB source.

Three LightGBM quantile regressors (0.25/0.5/0.75) for `price_ratio`, per tenant.
ALWAYS benchmarked against the empirical baseline (per-product quantiles of
comparable won lines); the empirical band stays the SERVED fallback unless the
fitted models beat it on coverage AND pinball. Feature set is ADAPTIVE (uses
whichever agreed features are present; never synthesizes). Band clamped to ±30%
of list price."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel
from sklearn.metrics import mean_absolute_error, mean_pinball_loss
from sklearn.model_selection import train_test_split

from maxxflow_core.errors import get_logger
from maxxflow_mlops.naming import registered_model_name
from maxxflow_mlops.registry import MLflowRegistry
from maxxflow_features.cleaning import clean_frame
from m1_quote.csv_common import RunLogger, auto_tune_quantile

log = get_logger("m1_quote.csv_price")
MODULE = "m1_quote_price"
TARGET = "price_ratio"
GROUP = "productID"
QUANTILES = (0.25, 0.50, 0.75)
CLAMP = (0.70, 1.30)

PRICE_NUMERIC = ["quantity", "unitPrice", "leadTimeDays", "contact_win_rate"]
PRICE_CATEG = ["productID", "materialSpec", "region"]
FINALIZED = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 30}


def _select_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    return ([c for c in PRICE_NUMERIC if c in df.columns],
            [c for c in PRICE_CATEG if c in df.columns])


class PriceBandModel(PythonModel):
    def __init__(self, regressors, categories, feature_columns, empirical, empirical_global,
                 served_mode, group_col, provenance="csv"):
        self.regressors = regressors
        self.categories = categories
        self.feature_columns = feature_columns
        self.empirical = empirical
        self.empirical_global = empirical_global
        self.served_mode = served_mode
        self.group_col = group_col
        self.provenance = provenance

    def predict(self, context, model_input, params=None):
        df = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        n = len(df)
        if self.served_mode == "fitted":
            X = df.copy()
            for c, cats in self.categories.items():
                X[c] = pd.Categorical(X.get(c), categories=cats)
            for c in self.feature_columns:
                if c not in self.categories:
                    X[c] = pd.to_numeric(X.get(c), errors="coerce").fillna(0.0)
            lo = self.regressors[0.25].predict(X[self.feature_columns])
            mid = self.regressors[0.50].predict(X[self.feature_columns])
            hi = self.regressors[0.75].predict(X[self.feature_columns])
        else:
            keys = df[self.group_col] if (self.group_col and self.group_col in df) else [None] * n
            bands = [self.empirical.get(str(k), self.empirical_global) for k in keys]
            lo = np.array([b[0] for b in bands]); mid = np.array([b[1] for b in bands]); hi = np.array([b[2] for b in bands])
        lo, mid, hi = np.sort(np.vstack([lo, mid, hi]), axis=0)
        lo, mid, hi = np.clip(lo, *CLAMP), np.clip(mid, *CLAMP), np.clip(hi, *CLAMP)
        unit = pd.to_numeric(df.get("unitPrice"), errors="coerce").fillna(0.0).to_numpy() if "unitPrice" in df else np.zeros(n)
        return pd.DataFrame({
            "ratio_low": np.round(lo, 4), "ratio_mid": np.round(mid, 4), "ratio_high": np.round(hi, 4),
            "recommended_price_low": np.round(unit * lo, 2),
            "recommended_price_mid": np.round(unit * mid, 2),
            "recommended_price_high": np.round(unit * hi, 2), "served_mode": self.served_mode})


def _emp_bands(df: pd.DataFrame, group_col: str | None):
    gq = df[TARGET].quantile(list(QUANTILES))
    glob = (float(gq[0.25]), float(gq[0.50]), float(gq[0.75]))
    if not group_col:
        return {}, glob
    g = df.groupby(df[group_col].astype(str))[TARGET].quantile(list(QUANTILES)).unstack()
    return {k: (float(r[0.25]), float(r[0.50]), float(r[0.75])) for k, r in g.iterrows()}, glob


def _pin(y, a, b, c):
    return float(np.mean([mean_pinball_loss(y, a, alpha=0.25), mean_pinball_loss(y, b, alpha=0.50),
                          mean_pinball_loss(y, c, alpha=0.75)]))


def train(df_or_path, tenant: str, *, auto_hpo: bool = True, register: bool = True,
          source: str = "csv", logger: RunLogger | None = None) -> dict:
    logger = logger or RunLogger()
    df = pd.read_csv(df_or_path) if isinstance(df_or_path, str) else df_or_path.copy()
    # `tenant` names the registered model only — never a feature or a row filter
    # (see csv_win.train). Train on the whole frame (pooled = GLOBAL base model).
    if "tenant" in df.columns:
        present = sorted(map(str, df["tenant"].dropna().unique()))
        logger.log(f"'tenant' column present {present} — ignored for modelling; training on "
                   f"all {len(df)} won lines (tenant only names the model '{tenant}')")
    else:
        logger.log(f"Loaded {len(df)} won lines (source={source}, model tenant={tenant})")
    if TARGET not in df.columns:
        raise ValueError(f"dataset missing required columns: ['{TARGET}']")
    num, cat = _select_features(df)
    features = num + cat
    dropped = [c for c in PRICE_NUMERIC + PRICE_CATEG if c not in features]
    if not features:
        raise ValueError("no usable price features present — cannot train")
    group_col = GROUP if GROUP in df.columns else None
    logger.log(f"Using {len(features)} features: {features}"
               + (f"  |  not present (skipped): {dropped}" if dropped else ""))

    # Generic pass (shared by all modules): dedupe -> winsorise -> impute.
    # The target is included so an absurd price_ratio can't skew the fit either.
    df, dq = clean_frame(df, num + [TARGET], cat, logger,
                         dedupe_subset=num + list(cat) + [TARGET])
    logger.log(f"Review & Clean: {dq.summary()}")
    y = pd.to_numeric(df[TARGET], errors="coerce").fillna(1.0)
    X = df[features].copy()
    for c in cat:
        X[c] = X[c].astype("category")
    grp = df[group_col].astype(str) if group_col else pd.Series(["_"] * len(df), index=df.index)
    Xtr, Xte, ytr, yte, gtr, _ = train_test_split(X, y, grp, test_size=0.3, random_state=7)
    logger.log(f"Split: {len(Xtr)} train / {len(Xte)} holdout")
    params = auto_tune_quantile(Xtr, ytr, FINALIZED, logger, quantiles=QUANTILES) if auto_hpo else dict(FINALIZED)

    regressors = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a in QUANTILES:
            regressors[a] = LGBMRegressor(objective="quantile", alpha=a, verbose=-1,
                                          random_state=7, **params).fit(Xtr, ytr)
    f25, f50, f75 = (regressors[a].predict(Xte) for a in QUANTILES)
    fit_cov = float(np.mean((yte.to_numpy() >= f25) & (yte.to_numpy() <= f75)))
    fit_pin, fit_mae = _pin(yte, f25, f50, f75), float(mean_absolute_error(yte, f50))
    logger.log(f"Fitted: coverage={fit_cov:.3f} pinball={fit_pin:.4f} P50-MAE={fit_mae:.4f}")

    emp_tr, emp_g = _emp_bands(pd.DataFrame({group_col or "_g": gtr, TARGET: ytr}), group_col)
    e = np.array([emp_tr.get(str(k), emp_g) for k in (df[group_col].astype(str).loc[Xte.index] if group_col else [None] * len(Xte))])
    e25, e50, e75 = e[:, 0], e[:, 1], e[:, 2]
    emp_cov, emp_pin = float(np.mean((yte.to_numpy() >= e25) & (yte.to_numpy() <= e75))), _pin(yte, e25, e50, e75)
    logger.log(f"Empirical: coverage={emp_cov:.3f} pinball={emp_pin:.4f}")

    served = "fitted" if (fit_cov >= emp_cov and fit_pin <= emp_pin) else "empirical"
    logger.log(f"Served band = {served.upper()} "
               f"({'fitted beats empirical' if served == 'fitted' else 'empirical retained as fallback'})")

    metrics = {"coverage": fit_cov, "pinball_mean": fit_pin, "mae_p50": fit_mae,
               "empirical_coverage": emp_cov, "empirical_pinball_mean": emp_pin,
               "n_train": int(len(Xtr)), "n_test": int(len(Xte))}
    categories = {c: list(X[c].cat.categories) for c in cat}
    emp_all, emp_all_g = _emp_bands(df[[c for c in [group_col, TARGET] if c]] if group_col else df[[TARGET]], group_col)
    model = PriceBandModel(regressors, categories, features, emp_all, emp_all_g, served, group_col, provenance=source)

    result = {"model_type": "regression", "metrics": metrics, "served_mode": served,
              "features": features, "dropped_features": dropped, "source": source,
              "params": params, "logs": logger.lines}
    if not register:
        result["_model"] = model
        return result

    reg = MLflowRegistry()
    name = registered_model_name(tenant, MODULE)
    tags = {"tenant": tenant, "module": MODULE, "data_provenance": source, "served_mode": served,
            "coverage": f"{fit_cov:.5f}", "pinball_mean": f"{fit_pin:.5f}", "mae_p50": f"{fit_mae:.5f}"}
    present_raw = [c for c in (PRICE_NUMERIC + PRICE_CATEG + ["unitPrice"]) if c in df.columns]
    present_raw = list(dict.fromkeys(present_raw))
    ex = df[present_raw].head(3).copy()
    for c in present_raw:
        ex[c] = ex[c].astype(str) if c in PRICE_CATEG else pd.to_numeric(ex[c], errors="coerce").astype(float)
    sig = infer_signature(ex, model.predict(None, ex))
    version = reg.log_and_register(model, name=name, params={"algo": "lightgbm-quantile", **params},
                                   metrics=metrics, tags=tags, signature=sig, input_example=ex)
    result.update({"registered_name": name, "version": version, "champion": _champion_metrics(reg, name)})
    logger.log(f"Registered {name} v{version} (candidate — not yet champion)")
    return result


def _champion_metrics(reg: MLflowRegistry, name: str) -> dict | None:
    v = reg.get_alias_version(name=name, alias="champion")
    if v is None:
        return None
    t = reg.get_alias_tags(name=name, alias="champion")
    return {"version": v, "coverage": float(t.get("coverage", 0)),
            "pinball_mean": float(t.get("pinball_mean", 1)), "served_mode": t.get("served_mode", "?")}


def publish(tenant: str, version: str, candidate_metrics: dict, force: bool = False) -> dict:
    reg = MLflowRegistry()
    name = registered_model_name(tenant, MODULE)
    champ = _champion_metrics(reg, name)
    if champ is None:
        promote, reasons = True, ["no incumbent champion — publish first model"]
    else:
        promote = (candidate_metrics["coverage"] >= champ["coverage"]
                   and candidate_metrics["pinball_mean"] <= champ["pinball_mean"])
        reasons = ([f"coverage {candidate_metrics['coverage']:.3f}≥{champ['coverage']:.3f} and "
                    f"pinball {candidate_metrics['pinball_mean']:.4f}≤{champ['pinball_mean']:.4f}"] if promote
                   else ["does not beat champion on coverage AND pinball — keep current"])
    if force and not promote:
        reasons = [f"forced publish — overriding gate: {reasons[0]}"] + reasons
        promote = True
    if promote:
        reg.promote(name=name, challenger_version=str(version))
    return {"published": promote, "reasons": reasons, "champion_before": champ, "candidate": candidate_metrics}
