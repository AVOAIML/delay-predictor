"""Shared helpers for the CSV-trained M1 models (win classifier + price band).

Provides: a real timestamped run-logger (streamed to the Configurator "Train" UI),
a light dataset clean (Review & Clean step), a small CPU-friendly random hyper-
parameter search with genuine per-trial logs, and required-column validation.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import KFold, cross_val_score
from maxxflow_features.cleaning import clean_frame as _shared_clean_frame

@dataclass
class RunLogger:
    """Collects real, timestamped training log lines (the Train page tails these)."""
    lines: list = field(default_factory=list)

    def log(self, msg: str) -> str:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.lines.append(line)
        return line

    def info(self, msg: str) -> str:
        return self.log(msg)


def require_columns(df: pd.DataFrame, needed: list[str]) -> list[str]:
    """Return the required columns that are MISSING (Review & Clean validation)."""
    return [c for c in needed if c not in df.columns]


def clean_frame(df: pd.DataFrame, numeric: list[str], categorical: list[str],
                logger: RunLogger | None = None, *, dedupe_subset=None,
                winsorize: bool = True, stats=None):
    """Generic Review & Clean pass — DEDUPE -> WINSORISE -> IMPUTE.

    Thin delegation to :func:`maxxflow_features.clean_frame` so all four modules share
    one implementation (the generic mechanics live in ``libs``; domain defaults stay in
    each module's ``features.py``). Returns ``(frame, CleaningReport)``.
    """
    return _shared_clean_frame(df, numeric, categorical, logger,
                               dedupe_subset=dedupe_subset, winsorize=winsorize, stats=stats)



# --- light auto-HPO for the win classifier (real per-trial CV logs) ---------
_SEARCH_SPACE = {
    "n_estimators": [200, 300, 400],
    "learning_rate": [0.03, 0.05, 0.08],
    "num_leaves": [15, 31, 63],
    "min_child_samples": [10, 20, 40],
}


def auto_tune_classifier(X: pd.DataFrame, y: pd.Series, finalized: dict,
                         logger: RunLogger, n_iter: int = 8, cv: int = 3) -> dict:
    """Small random search around the finalized params, scored by CV ROC-AUC.
    The finalized config is always a candidate; we keep it unless a search config
    genuinely beats it (so 'finalized' is the anchor, HPO only improves)."""
    rng = random.Random(7)
    candidates = [("finalized", dict(finalized))]
    for i in range(n_iter):
        candidates.append((f"search-{i+1}",
                           {k: rng.choice(v) for k, v in _SEARCH_SPACE.items()}))
    best = None
    for name, params in candidates:
        try:
            est = LGBMClassifier(verbose=-1, random_state=7, **params)
            score = float(cross_val_score(est, X, y, cv=cv, scoring="roc_auc").mean())
        except Exception as e:  # degenerate fold etc. — skip this config
            logger.log(f"HPO {name}: skipped ({type(e).__name__})")
            continue
        logger.log(f"HPO {name}: CV AUC={score:.4f}  "
                   f"(n={params['n_estimators']}, lr={params['learning_rate']}, "
                   f"leaves={params['num_leaves']}, mcs={params['min_child_samples']})")
        if best is None or score > best[0]:
            best = (score, name, params)
    chosen = best[2] if best else dict(finalized)
    logger.log(f"HPO best: {best[1] if best else 'finalized'} @ CV AUC={best[0]:.4f}"
               if best else "HPO: fell back to finalized params")
    return chosen


# --- light auto-HPO for the price-band quantile regressors (real per-trial CV logs) --
_REG_SEARCH_SPACE = {
    "n_estimators": [200, 300, 400, 600],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "num_leaves": [15, 31, 63],
    "min_child_samples": [10, 20, 30, 50],
}


def auto_tune_quantile(X: pd.DataFrame, y: pd.Series, finalized: dict, logger: RunLogger,
                       quantiles=(0.25, 0.50, 0.75), n_iter: int = 8, cv: int = 3) -> dict:
    """Small random search around the finalized params, scored by mean CV pinball loss
    across all three quantiles (one shared param set fits all three regressors — only
    `alpha` differs between them, same as the final fit). The finalized config is
    always a candidate; we keep it unless a search config genuinely beats it."""
    rng = random.Random(7)
    candidates = [("finalized", dict(finalized))]
    for i in range(n_iter):
        candidates.append((f"search-{i+1}",
                           {k: rng.choice(v) for k, v in _REG_SEARCH_SPACE.items()}))
    kf = KFold(n_splits=cv, shuffle=True, random_state=7)
    best = None
    for name, params in candidates:
        try:
            fold_pins = []
            for tr_idx, va_idx in kf.split(X):
                Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
                ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
                preds = {}
                for a in quantiles:
                    m = LGBMRegressor(objective="quantile", alpha=a, verbose=-1,
                                      random_state=7, **params).fit(Xtr, ytr)
                    preds[a] = m.predict(Xva)
                fold_pins.append(float(np.mean([mean_pinball_loss(yva, preds[a], alpha=a)
                                                for a in quantiles])))
        except Exception as e:  # degenerate fold etc. — skip this config
            logger.log(f"HPO {name}: skipped ({type(e).__name__})")
            continue
        score = float(np.mean(fold_pins))
        logger.log(f"HPO {name}: CV pinball={score:.4f}  "
                   f"(n={params['n_estimators']}, lr={params['learning_rate']}, "
                   f"leaves={params['num_leaves']}, mcs={params['min_child_samples']})")
        if best is None or score < best[0]:
            best = (score, name, params)
    chosen = best[2] if best else dict(finalized)
    logger.log(f"HPO best: {best[1]} @ CV pinball={best[0]:.4f}"
               if best else "HPO: fell back to finalized params")
    return chosen


# --- sample-size confidence (per-product win model) -------------------------
# Reuses the BRD "hide if <5 comparable historical quotes" boundary from
# m1_quote.model.MIN_COMPARABLE, so "few historical quotes -> Low confidence"
# stays consistent with that existing guardrail rather than a new magic number.
def confidence_level(n_comparable: int) -> str:
    from m1_quote.model import MIN_COMPARABLE
    if n_comparable < MIN_COMPARABLE:
        return "Low"
    if n_comparable < 4 * MIN_COMPARABLE:
        return "Medium"
    return "High"


def as_category(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype("object").astype("category")
    return out
