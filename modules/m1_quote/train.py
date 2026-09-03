"""M1 training (plan §4 M1, §6). LightGBM + isotonic calibration so the % is a
true probability; price bands = quantiles over comparable WON quotes; register by
NAME with metrics as version tags; champion/challenger gate moves ``@champion``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from maxxflow_core.clock import get_clock
from maxxflow_core.errors import get_logger
from maxxflow_mlops.naming import registered_model_name
from maxxflow_mlops.promotion import (expected_calibration_error, gate_metrics,
                                       should_promote)
from maxxflow_mlops.registry import MLflowRegistry
from m1_quote.features import CATEGORICAL, FEATURE_COLUMNS, build_features, training_frame
from m1_quote.model import QuoteModel

log = get_logger("m1_quote.train")
MODULE = "m1_quote"


def _price_bands(tables: dict, feature_frame: pd.DataFrame) -> dict:
    """Per product-type quantile band over WON quotes' per-unit sales prices."""
    won_ids = set(feature_frame.loc[feature_frame["label"] == 1, "id"])
    li = tables["quotation_line_items"]
    # only product_type_id from products; the band uses the LINE's negotiated sales_price
    prod = tables["products"][["id", "product_type_id"]].rename(columns={"id": "product_id"})
    won_lines = li[li["quotation_id"].isin(won_ids)].merge(prod, on="product_id", how="left")
    bands = {}
    if not won_lines.empty:
        # map product_type_id -> code using the products table + a reverse lookup is
        # unnecessary; group by the type id then translate via feature_frame mapping.
        from m1_quote.features import _product_type_reverse  # local import to avoid cycle at import time
        md = feature_frame.attrs.get("md")
        rev = _product_type_reverse(md) if md is not None else {}
        won_lines["ptype"] = won_lines["product_type_id"].map(rev).fillna("FINISHED_GOOD")
        for ptype, grp in won_lines.groupby("ptype"):
            q = grp["sales_price"].astype(float).quantile([0.25, 0.5, 0.75]).tolist()
            bands[ptype] = (round(q[0], 2), round(q[1], 2), round(q[2], 2))
        allq = won_lines["sales_price"].astype(float).quantile([0.25, 0.5, 0.75]).tolist()
        bands["__global__"] = (round(allq[0], 2), round(allq[1], 2), round(allq[2], 2))
    return bands


def fit_model(feature_frame: pd.DataFrame, tables: dict | None, seed: int = 7):
    X, y = training_frame(feature_frame)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        booster = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                 min_child_samples=20, verbose=-1, random_state=seed)
        booster.fit(Xtr, ytr, categorical_feature=CATEGORICAL)
    raw_te = booster.predict_proba(Xte)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_te, yte)
    cal_te = np.clip(iso.predict(raw_te), 0, 1)

    pred_te = (cal_te >= 0.5).astype(int)
    # accuracy of always guessing the majority class on this holdout — what the
    # promotion floor's skill check measures accuracy against.
    base_rate = float(max(yte.mean(), 1 - yte.mean()))
    accuracy = float((pred_te == np.asarray(yte)).mean())
    metrics = {
        "auc": float(roc_auc_score(yte, cal_te)),
        "brier": float(brier_score_loss(yte, cal_te)),
        "ece": float(expected_calibration_error(yte, cal_te)),
        "accuracy": accuracy,
        "base_rate_accuracy": base_rate,
        "accuracy_over_base_rate": accuracy - base_rate,
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "positive_rate": float(y.mean()),
    }
    categories = {c: list(X[c].cat.categories) for c in CATEGORICAL if hasattr(X[c], "cat")}
    bands = _price_bands(tables, feature_frame) if tables else {}
    model = QuoteModel(booster=booster, calibrator=iso, categories=categories,
                       price_bands=bands, provenance="synthetic")
    return model, metrics


def _example(feature_frame: pd.DataFrame) -> pd.DataFrame:
    ex = feature_frame[FEATURE_COLUMNS].head(3).copy()
    # serving contract uses plain string for the category (the model re-casts
    # internally); float for the int feature (mlflow: ints can't hold missing).
    ex["product_type_code"] = ex["product_type_code"].astype(str)
    ex["n_comparable"] = feature_frame["n_comparable"].head(3).astype(float).to_numpy()
    ex["base_price"] = 1000.0
    return ex


def train(tenant: str = "demo", *, tables: dict | None = None, md=None,
          feature_frame: pd.DataFrame | None = None, register: bool = True, seed: int = 7):
    clock = get_clock()
    if feature_frame is None:
        if tables is None:
            from m1_quote.dal import read_quote_tables
            tables, md = read_quote_tables(tenant)
        feature_frame = build_features(tables, md, clock)
    if md is not None:
        feature_frame.attrs["md"] = md

    model, metrics = fit_model(feature_frame, tables, seed=seed)
    log.info("trained M1 auc=%.3f brier=%.3f ece=%.3f", metrics["auc"], metrics["brier"], metrics["ece"])
    if not register:
        return model, metrics

    reg = MLflowRegistry()
    name = registered_model_name(tenant, MODULE)
    tags = {"tenant": tenant, "module": MODULE, "data_provenance": "synthetic",
            "auc": f"{metrics['auc']:.5f}", "brier": f"{metrics['brier']:.5f}", "ece": f"{metrics['ece']:.5f}"}
    sig = infer_signature(_example(feature_frame), model.predict(None, _example(feature_frame)))
    version = reg.log_and_register(model, name=name, params={"algo": "lightgbm+isotonic", "seed": seed},
                                   metrics=metrics, tags=tags, signature=sig,
                                   input_example=_example(feature_frame))

    champ_v = reg.get_alias_version(name=name, alias="champion")
    champ_metrics = None
    if champ_v is not None:
        t = reg.get_alias_tags(name=name, alias="champion")
        champ_metrics = {"auc": float(t.get("auc", 0)), "brier": float(t.get("brier", 1))}
    decision = should_promote(
        gate_metrics(metrics),
        champ_metrics)
    if decision.promote:
        reg.promote(name=name, challenger_version=version)
        log.info("PROMOTED v%s to @champion: %s", version, "; ".join(decision.reasons))
    else:
        log.info("kept champion v%s (challenger v%s not promoted): %s", champ_v, version, "; ".join(decision.reasons))
    return version
