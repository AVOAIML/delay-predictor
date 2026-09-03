"""The four latent ground-truth functions (plan §4 cold-start signals, §5).

Centralised so the simulators and the learnability gate agree on what "truth" is.
Each function maps standardized drivers -> a probability (or rate), with a tunable
``noise`` that sets the achievable signal-to-noise — calibrated so a fair model
lands inside the module's acceptance band (NOT AUC≈1.0, which is a leak alarm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else x * 0.0


@dataclass(frozen=True)
class M1Weights:
    bias: float = -0.15
    margin: float = 1.15          # higher margin (price) -> lower win; sign applied below
    value: float = -0.35          # bigger deals slightly harder to win
    propensity: float = 1.30      # contact's historical win propensity
    days_left: float = 0.45       # more validity runway -> slightly higher win
    skill: float = 0.65           # salesperson skill tier
    noise: float = 0.95


def m1_win_logit(margin_frac, log_value, contact_propensity, days_left_frac, skill_tier,
                 rng: np.random.Generator, w: M1Weights = M1Weights()) -> np.ndarray:
    """σ(w·[margin, value, contact_propensity, days_left, skill] + noise)."""
    z = (
        w.bias
        - w.margin * _z(margin_frac)          # fatter margin = higher price = lower win
        + w.value * _z(log_value)
        + w.propensity * (np.asarray(contact_propensity) - 0.5) * 2.0
        + w.days_left * _z(days_left_frac)
        + w.skill * _z(skill_tier)
        + w.noise * rng.standard_normal(len(margin_frac))
    )
    return z


@dataclass(frozen=True)
class M3Weights:
    bias: float = -1.05
    pace: float = 1.30            # slow pace-so-far at 25% milestone
    load: float = 1.05            # work-center overload (concurrency / capacity)
    skill: float = 0.95           # junior operator tier -> slower
    material: float = 1.20        # material shortfall
    complexity: float = 0.55      # ops + components + dependency depth
    noise: float = 1.55


def m3_delay_logit(pace_ratio, load_ratio, skill_tier, material_shortfall, complexity,
                   rng: np.random.Generator, w: M3Weights = M3Weights()) -> np.ndarray:
    z = (
        w.bias
        + w.pace * _z(pace_ratio)
        + w.load * _z(load_ratio)
        - w.skill * _z(skill_tier)            # higher tier (faster) -> less delay
        + w.material * _z(material_shortfall)
        + w.complexity * _z(complexity)
        + w.noise * rng.standard_normal(len(pace_ratio))
    )
    return z


@dataclass(frozen=True)
class M2Weights:
    bias: float = -0.35
    deficit: float = 1.45         # on-hand below ROP
    rate: float = 1.10            # high consumption rate vs cover
    cover: float = -1.25          # incoming PO coverage reduces risk
    reliability: float = -0.55    # reliable vendor reduces risk
    variability: float = 0.55     # demand variability
    noise: float = 1.15


def m2_stockout_logit(deficit, consumption_rate, incoming_cover, vendor_reliability, demand_var,
                      rng: np.random.Generator, w: M2Weights = M2Weights()) -> np.ndarray:
    z = (
        w.bias
        + w.deficit * _z(deficit)
        + w.rate * _z(consumption_rate)
        + w.cover * _z(incoming_cover)
        + w.reliability * _z(vendor_reliability)
        + w.variability * _z(demand_var)
        + w.noise * rng.standard_normal(len(deficit))
    )
    return z
