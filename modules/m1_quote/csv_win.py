"""M1 Smart Quote Optimiser — WIN model (classification), CSV or DB source.

LightGBM + isotonic calibration, per tenant, with light auto-HPO. The feature set
is ADAPTIVE: it uses whichever agreed features are actually present in the source
(CSV or read-replica) and never synthesises missing ones — so it degrades cleanly
when the DB lacks columns like region/industry (plan: "add them back when the
columns appear; do NOT synthesize"). Registers a candidate; publish() moves the
champion alias through the gate."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from maxxflow_core.errors import get_logger
from maxxflow_mlops.naming import registered_model_name
from maxxflow_mlops.promotion import (expected_calibration_error, gate_metrics,
                                       should_promote)
from maxxflow_mlops.registry import MLflowRegistry
from maxxflow_features.cleaning import clean_frame
from m1_quote.csv_common import RunLogger, auto_tune_classifier

log = get_logger("m1_quote.csv_win")
MODULE = "m1_quote_win"
LABEL = "won"

# `log_value` is derived from grand_total; the rest are read directly.
WIN_NUMERIC = ["log_value", "total_quantity", "line_count", "n_products", "wtd_price_ratio",
               "mean_price_ratio", "min_price_ratio", "avg_discount_pct",
               "contact_win_rate", "salesrep_win_rate"]
WIN_CATEG = ["region", "industry"]
RAW_INPUT = ["grand_total", "total_quantity", "line_count", "n_products", "wtd_price_ratio",
             "mean_price_ratio", "min_price_ratio", "avg_discount_pct",
             "contact_win_rate", "salesrep_win_rate", "region", "industry"]
FINALIZED = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20}


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "log_value" not in out.columns and "grand_total" in out.columns:
        out["log_value"] = np.log1p(pd.to_numeric(out["grand_total"], errors="coerce").clip(lower=0))
    return out


def _select_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features) actually available."""
    num = [c for c in WIN_NUMERIC
           if (c == "log_value" and "grand_total" in df.columns) or (c != "log_value" and c in df.columns)]
    cat = [c for c in WIN_CATEG if c in df.columns]
    return num, cat


class WinModel(PythonModel):
    def __init__(self, booster, calibrator, categories: dict, feature_columns: list[str], provenance="csv"):
        self.booster = booster
        self.calibrator = calibrator
        self.categories = categories
        self.feature_columns = feature_columns
        self.provenance = provenance

    def predict(self, context, model_input, params=None):
        df = _prep(model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input))
        for c, cats in self.categories.items():
            df[c] = pd.Categorical(df.get(c), categories=cats)
        for c in self.feature_columns:
            if c not in self.categories:
                df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0.0)
        raw = self.booster.predict_proba(df[self.feature_columns])[:, 1]
        p = np.clip(self.calibrator.predict(raw), 0.0, 1.0)
        return pd.DataFrame({"win_probability": np.round(p, 4),
                             "win_probability_pct": np.round(p * 100, 1),
                             "low_confidence": (p < 0.40) | (p > 0.95)})


def train(df_or_path, tenant: str, *, auto_hpo: bool = True, register: bool = True,
          source: str = "csv", logger: RunLogger | None = None) -> dict:
    logger = logger or RunLogger()
    df = pd.read_csv(df_or_path) if isinstance(df_or_path, str) else df_or_path.copy()
    # `tenant` is NOT a feature and NOT a row filter: it only names the registered
    # model (t_<tenant>__m_...). We train on the WHOLE frame so a pooled dataset can
    # build the shared GLOBAL base model, and a tenant's own export trains on all its
    # rows regardless of any internal `tenant` label.
    if "tenant" in df.columns:
        present = sorted(map(str, df["tenant"].dropna().unique()))
        logger.log(f"'tenant' column present {present} — ignored for modelling; training on "
                   f"all {len(df)} rows (tenant only names the model '{tenant}')")     
    else:
        logger.log(f"Loaded {len(df)} rows (source={source}, model tenant={tenant})")
        logger.info(f"Loaded {len(df)} rows (source={source}, model tenant={tenant})")

    if LABEL not in df.columns:
        raise ValueError(f"dataset missing required columns: ['{LABEL}']")
    num, cat = _select_features(df)
    features = num + cat
    dropped = [c for c in WIN_NUMERIC + WIN_CATEG if c not in features]
    if len(features) < 2:
        raise ValueError(f"only {len(features)} usable feature(s) present — cannot train")
    logger.log(f"Using {len(features)} features: {features}"
               + (f"  |  not present (skipped, not synthesized): {dropped}" if dropped else ""))

    df = _prep(df)
    # Generic pass (shared by all modules): dedupe -> winsorise -> impute.
    # Dedupe on features + label: identical features with DIFFERENT outcomes are real
    # label noise and must be kept, so the label is part of the key.
    _num_for_clean = [c for c in num if c != "log_value"] + (["log_value"] if "log_value" in num else [])
    df, dq = clean_frame(df, _num_for_clean, cat, logger,
                         dedupe_subset=_num_for_clean + list(cat) + [LABEL])
    logger.log(f"Review & Clean: {dq.summary()}")

    y = pd.to_numeric(df[LABEL], errors="coerce").fillna(0).astype(int)
    if y.nunique() < 2:
        raise ValueError(f"tenant {tenant}: only one class present — cannot train a classifier")
    X = df[features].copy()
    for c in cat:
        X[c] = X[c].astype("category")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)
    logger.log(f"Split: {len(Xtr)} train / {len(Xte)} holdout")
    params = auto_tune_classifier(Xtr, ytr, FINALIZED, logger) if auto_hpo else dict(FINALIZED)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        booster = LGBMClassifier(verbose=-1, random_state=7, **params).fit(Xtr, ytr)
    iso = IsotonicRegression(out_of_bounds="clip").fit(booster.predict_proba(Xte)[:, 1], yte)
    p = np.clip(iso.predict(booster.predict_proba(Xte)[:, 1]), 0, 1)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    # accuracy of always guessing the majority class on THIS holdout — the number
    # the promotion floor's skill check measures accuracy against.
    base_rate = float(max(yte.mean(), 1 - yte.mean()))
    metrics = {
        "accuracy": float((pred == yte.to_numpy()).mean()),
        "base_rate_accuracy": base_rate,
        "accuracy_over_base_rate": float((pred == yte.to_numpy()).mean() - base_rate),
        "auc": float(roc_auc_score(yte, p)), "pr_auc": float(average_precision_score(yte, p)),
        "precision": float(precision_score(yte, pred, zero_division=0)),
        "recall": float(recall_score(yte, pred, zero_division=0)),
        "f1": float(f1_score(yte, pred, zero_division=0)),
        "brier": float(brier_score_loss(yte, p)), "ece": float(expected_calibration_error(yte, p)),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)), "positive_rate": float(y.mean()),
    }
    logger.log(f"Metrics: accuracy={metrics['accuracy']:.3f} AUC={metrics['auc']:.3f} "
               f"F1={metrics['f1']:.3f} Brier={metrics['brier']:.3f} ECE={metrics['ece']:.3f}")
    categories = {c: list(X[c].cat.categories) for c in cat}
    model = WinModel(booster, iso, categories, features, provenance=source)

    result = {"model_type": "classification", "metrics": metrics, "features": features,
              "dropped_features": dropped, "source": source,
              "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
              "params": params, "logs": logger.lines}
    if not register:
        result["_model"] = model
        return result

    reg = MLflowRegistry()
    name = registered_model_name(tenant, MODULE)
    tags = {"tenant": tenant, "module": MODULE, "data_provenance": source,
            "accuracy": f"{metrics['accuracy']:.5f}", "auc": f"{metrics['auc']:.5f}",
            "brier": f"{metrics['brier']:.5f}", "ece": f"{metrics['ece']:.5f}"}
    present_raw = [c for c in RAW_INPUT if c in df.columns]
    ex = df[present_raw].head(3).copy()
    for c in present_raw:
        ex[c] = ex[c].astype(str) if c in WIN_CATEG else pd.to_numeric(ex[c], errors="coerce").astype(float)
    sig = infer_signature(ex, model.predict(None, ex))
    version = reg.log_and_register(model, name=name, params={"algo": "lightgbm+isotonic", **params},
                                   metrics=metrics, tags=tags, signature=sig, input_example=ex)
    logger.log(f"Registered {name} v{version} (candidate — not yet champion)")
    result.update({"registered_name": name, "version": version, "champion": _champion_metrics(reg, name)})
    return result


def _champion_metrics(reg: MLflowRegistry, name: str) -> dict | None:
    v = reg.get_alias_version(name=name, alias="champion")
    if v is None:
        return None
    t = reg.get_alias_tags(name=name, alias="champion")
    return {"version": v, "accuracy": float(t.get("accuracy", 0)), "auc": float(t.get("auc", 0)),
            "brier": float(t.get("brier", 1)), "ece": float(t.get("ece", 1))}


def publish(tenant: str, version: str, candidate_metrics: dict, force: bool = False) -> dict:
    reg = MLflowRegistry()
    name = registered_model_name(tenant, MODULE)
    champ = _champion_metrics(reg, name)
    decision = should_promote(
        gate_metrics(candidate_metrics),
        None if champ is None else {"auc": champ["auc"], "brier": champ["brier"]}, force=force)
    if decision.promote:
        reg.promote(name=name, challenger_version=str(version))
    return {"published": decision.promote, "reasons": decision.reasons,
            # the single DECIDING check when refused. A caller showing reasons[0]
            # next to a refusal could otherwise quote a check that passed.
            "blocker": decision.blocker, "gate_checks": decision.checks,
            "gate_summary": decision.summary,
            "champion_before": champ, "candidate": candidate_metrics}
