"""M1 served model (plan §4 M1). A uniform ``pyfunc.PythonModel`` (same flavour as
the parity stub and every other module) wrapping a LightGBM classifier + isotonic
calibrator, returning calibrated win probability + a recommended price band, with
ALL BRD guardrails applied in-model:

* hide if < 5 comparable historical quotes
* low-confidence flag if score < 40% or > 95%
* price band clamped to ±30% of base ``Product.salesPrice`` (decimal.Decimal)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel

from maxxflow_core.money import clamp_price
from m1_quote.features import CATEGORICAL, FEATURE_COLUMNS

MIN_COMPARABLE = 5      # hide below this many comparable historical quotes
LOW_CONF_LOW = 0.40     # low-confidence band edges
LOW_CONF_HIGH = 0.95


class QuoteModel(PythonModel):
    def __init__(self, booster, calibrator, categories: dict, price_bands: dict,
                 provenance: str = "synthetic"):
        self.booster = booster
        self.calibrator = calibrator
        self.categories = categories
        self.price_bands = price_bands           # ptype_code -> (low, mid, high)
        self.global_band = price_bands.get("__global__", (None, None, None))
        self.provenance = provenance

    # -- pyfunc contract ----------------------------------------------------
    def predict(self, context, model_input, params=None):
        df = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        X = df.copy()
        for col in CATEGORICAL:
            cats = self.categories.get(col)
            if cats is not None and col in X.columns:
                X[col] = pd.Categorical(X[col], categories=cats)
        raw = self.booster.predict_proba(X[FEATURE_COLUMNS])[:, 1]
        prob = np.clip(self.calibrator.predict(raw), 0.0, 1.0)

        n = len(df)
        has_ncomp = "n_comparable" in df.columns
        has_base = "base_price" in df.columns
        ptypes = df["product_type_code"] if "product_type_code" in df.columns else pd.Series([None] * n)

        out = []
        for i in range(n):
            p = float(prob[i])
            low_conf = (p < LOW_CONF_LOW) or (p > LOW_CONF_HIGH)
            ncomp = int(df["n_comparable"].iloc[i]) if has_ncomp and pd.notna(df["n_comparable"].iloc[i]) else None
            hidden = (ncomp is not None and ncomp < MIN_COMPARABLE)
            band = self.price_bands.get(ptypes.iloc[i], self.global_band)
            lo, mid, hi = band
            clamped = False
            if has_base and pd.notna(df["base_price"].iloc[i]):
                base = df["base_price"].iloc[i]
                lo2 = float(clamp_price(lo, base)) if lo is not None else None
                hi2 = float(clamp_price(hi, base)) if hi is not None else None
                clamped = (lo2 != lo) or (hi2 != hi)
                lo, hi = lo2, hi2
            out.append({
                "win_probability": round(p, 4),
                "win_probability_pct": round(p * 100, 1),
                "low_confidence": bool(low_conf),
                "hidden": bool(hidden),
                "n_comparable": ncomp,
                "recommended_price_low": lo,
                "recommended_price_mid": mid,
                "recommended_price_high": hi,
                "price_clamped": bool(clamped),
                "model_provenance": self.provenance,
            })
        return pd.DataFrame(out)
