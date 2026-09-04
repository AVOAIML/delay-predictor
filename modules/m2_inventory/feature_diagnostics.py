"""Numeric feature-support, drift, correlation, and OOD diagnostics for M2."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


NEAR_ZERO_IMPUTED_STD = 0.005
EXTREME_Z_THRESHOLD = 20.0
HIGH_CORRELATION_THRESHOLD = 0.95


def audit_numeric_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_features: list[str],
    *,
    synthetic_column: str = "business_scenario_injected",
) -> pd.DataFrame:
    """Describe training support and untouched-test drift without fitting on test."""
    synthetic = train.get(synthetic_column, pd.Series(False, index=train.index))
    synthetic = synthetic.astype("boolean").fillna(False).astype(bool)
    rows: list[dict[str, float | int | str | bool]] = []
    for feature in numeric_features:
        train_values = pd.to_numeric(train[feature], errors="coerce")
        test_values = pd.to_numeric(test[feature], errors="coerce")
        observed_train = train_values.dropna()
        observed_test = test_values.dropna()
        median = float(observed_train.median()) if len(observed_train) else 0.0
        imputed_train = train_values.fillna(median)
        imputed_mean = float(imputed_train.mean())
        imputed_std = float(imputed_train.std(ddof=0))
        imputed_test = test_values.fillna(median)
        if imputed_std > 0.0:
            max_abs_z = float(((imputed_test - imputed_mean) / imputed_std).abs().max())
            standardized_mean_shift = float(
                abs(float(imputed_test.mean()) - imputed_mean) / imputed_std
            )
        else:
            max_abs_z = float("inf") if imputed_test.ne(imputed_mean).any() else 0.0
            standardized_mean_shift = max_abs_z
        real = train_values[~synthetic].dropna()
        generated = train_values[synthetic].dropna()
        real_nonzero_rows = int(real.ne(0.0).sum())
        synthetic_nonzero_rows = int(generated.ne(0.0).sum())
        total_nonzero_rows = real_nonzero_rows + synthetic_nonzero_rows
        if len(observed_train) and len(observed_test):
            ks = float(ks_2samp(observed_train, observed_test).statistic)
        else:
            ks = float("nan")
        q1 = float(observed_train.quantile(0.25)) if len(observed_train) else float("nan")
        q3 = float(observed_train.quantile(0.75)) if len(observed_train) else float("nan")
        rows.append(
            {
                "feature": feature,
                "train_min": observed_train.min(),
                "train_max": observed_train.max(),
                "train_median": median,
                "train_mean": observed_train.mean(),
                "train_std": observed_train.std(ddof=0),
                "train_iqr": q3 - q1,
                "train_pct_zero": float(train_values.eq(0.0).mean() * 100.0),
                "train_pct_missing": float(train_values.isna().mean() * 100.0),
                "test_min": observed_test.min(),
                "test_max": observed_test.max(),
                "max_abs_test_z": max_abs_z,
                "imputed_train_mean": imputed_mean,
                "imputed_train_std": imputed_std,
                "standardized_mean_shift": standardized_mean_shift,
                "ks_statistic": ks,
                "real_rows": int(len(real)),
                "real_nonzero_rows": real_nonzero_rows,
                "real_mean": real.mean(),
                "real_std": real.std(ddof=0),
                "real_min": real.min(),
                "real_max": real.max(),
                "synthetic_rows": int(len(generated)),
                "synthetic_nonzero_rows": synthetic_nonzero_rows,
                "synthetic_share_nonzero": (
                    synthetic_nonzero_rows / total_nonzero_rows
                    if total_nonzero_rows else 0.0
                ),
                "synthetic_mean": generated.mean(),
                "synthetic_std": generated.std(ddof=0),
                "synthetic_min": generated.min(),
                "synthetic_max": generated.max(),
                "near_zero_variance": bool(imputed_std < NEAR_ZERO_IMPUTED_STD),
                "low_observed_support": bool(train_values.notna().mean() < 0.01),
                "synthetic_only_variation": bool(
                    real_nonzero_rows == 0 and synthetic_nonzero_rows > 0
                ),
                "synthetic_dominates_nonzero": bool(
                    total_nonzero_rows > 0
                    and synthetic_nonzero_rows / total_nonzero_rows > 0.5
                ),
                "extreme_test_ood": bool(max_abs_z > EXTREME_Z_THRESHOLD),
                "train_test_mismatch": bool(
                    standardized_mean_shift > 2.0 or (np.isfinite(ks) and ks > 0.5)
                ),
            }
        )
    return pd.DataFrame(rows)


def high_numeric_correlations(
    train: pd.DataFrame,
    numeric_features: list[str],
    threshold: float = HIGH_CORRELATION_THRESHOLD,
) -> pd.DataFrame:
    values = train[numeric_features].apply(pd.to_numeric, errors="coerce")
    matrix = values.corr(method="spearman")
    rows = []
    for left_index, left in enumerate(numeric_features):
        for right in numeric_features[left_index + 1 :]:
            correlation = matrix.loc[left, right]
            if pd.notna(correlation) and abs(float(correlation)) >= threshold:
                rows.append(
                    {
                        "feature_left": left,
                        "feature_right": right,
                        "spearman": float(correlation),
                        "absolute_spearman": abs(float(correlation)),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        "absolute_spearman", ascending=False, ignore_index=True
    ) if rows else pd.DataFrame(
        columns=["feature_left", "feature_right", "spearman", "absolute_spearman"]
    )


def enforce_training_safeguards(
    audit: pd.DataFrame, *, fail_on_zero_variance: bool = False
) -> list[str]:
    """Warn on unsafe support, with an opt-in strict zero-variance failure."""
    messages: list[str] = []
    unusable = audit.loc[audit["imputed_train_std"].le(1e-12), "feature"].tolist()
    if unusable:
        message = f"Numeric model features have zero training variance: {unusable}"
        if fail_on_zero_variance:
            raise ValueError(message)
        warnings.warn(f"M2 feature safeguard: {message}", RuntimeWarning, stacklevel=2)
        messages.append(message)
    for flag, label in (
        ("near_zero_variance", "near-zero imputed variance"),
        ("low_observed_support", "less than 1% observed training support"),
        ("extreme_test_ood", f"untouched-test |z| above {EXTREME_Z_THRESHOLD:g}"),
        ("train_test_mismatch", "large train/test distribution mismatch"),
        ("synthetic_only_variation", "non-zero variation exists only in synthetic rows"),
    ):
        features = audit.loc[audit[flag], "feature"].tolist()
        if features:
            message = f"M2 feature safeguard: {label}: {features}"
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            messages.append(message)
    return messages


def serving_ood_report(
    frame: pd.DataFrame,
    audit: pd.DataFrame,
    *,
    threshold: float = EXTREME_Z_THRESHOLD,
) -> pd.DataFrame:
    rows = []
    indexed = audit.set_index("feature")
    for feature in indexed.index:
        values = pd.to_numeric(frame[feature], errors="coerce")
        mean = float(indexed.at[feature, "imputed_train_mean"])
        std = float(indexed.at[feature, "imputed_train_std"])
        median = float(indexed.at[feature, "train_median"])
        imputed = values.fillna(median)
        if std <= 1e-12:
            z = pd.Series(
                np.where(imputed.eq(mean), 0.0, float("inf")), index=values.index
            )
        else:
            z = (imputed - mean) / std
        extreme_z = z.abs().gt(threshold)
        synthetic_only = bool(
            indexed.at[feature, "synthetic_only_variation"]
        ) if "synthetic_only_variation" in indexed else False
        synthetic_support = values.fillna(0.0).ne(0.0) & synthetic_only
        mask = extreme_z | synthetic_support
        if mask.any():
            rows.append(
                {
                    "feature": feature,
                    "rows": int(mask.sum()),
                    "max_abs_z": float(z[mask].abs().max()),
                    "min_value": float(values[mask].min()),
                    "max_value": float(values[mask].max()),
                    "extreme_z_rows": int(extreme_z.sum()),
                    "synthetic_only_support_rows": int(synthetic_support.sum()),
                }
            )
    return pd.DataFrame(rows)
