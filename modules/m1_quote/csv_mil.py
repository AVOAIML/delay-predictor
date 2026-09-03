"""M1 Smart Quote Optimiser — Noisy-OR Multiple Instance Learning (MIL) model.

Ports `modules/m1_quote/quotation_ml_mil.ipynb`'s idea to this repo's "classical
ML only, no torch in-process" constraint (CLAUDE.md / pyproject.toml /
maxxflow_mlops.serving): the network, its forward/backward pass, and its Adam
optimizer are hand-written in numpy — no PyTorch, no autodiff framework.

The core idea (unchanged from the notebook): a quotation's outcome is only known
at the QUOTE level, never per line, so each quote is a BAG of its line-item
INSTANCES. A small instance-level network predicts each line's own latent
acceptance probability p_i; the bag (quote) probability is the Noisy-OR
(product) of its instances: `P(quote won) = prod_i p_i`. Backprop on the
bag-level BCE loss distributes gradient signal down to every instance,
teaching the model which line looks bad without ever seeing a real per-line
label — unlike m1_quote_line_win, which broadcasts the quote label onto every
line as if it were a true per-line label.

Gradient derivation (BCE loss, yhat = prod_i p_i, p_i = sigmoid(z3_i)):
    dL/dyhat   = (yhat - y) / (yhat * (1 - yhat))              [standard BCE]
    dyhat/dp_j = yhat / p_j                                    [product rule]
    dL/dp_j    = (yhat - y) / (p_j * (1 - yhat))
    dp_j/dz3_j = p_j * (1 - p_j)                                [sigmoid]
    dL/dz3_j   = (yhat - y) * (1 - p_j) / (1 - yhat)           [the p_j cancels]
The clean final form is what `_noisy_or_grad` computes directly, verified
against finite differences in tests/unit/test_m1_mil_guardrails.py.

Two confidence signals are reported, same "two independent questions"
precedent as m1_quote_line_win: `confidence` (Low/Medium/High from
`n_comparable`, the PRODUCT's historical row count) and
`mc_dropout_confidence` (0-1, from MC-Dropout — T stochastic forward passes
with dropout kept active at inference, converting their spread into a
confidence score; tight spread -> high confidence).

This model does NOT serve its own price-band recommendation. `recommend_price_band`
still runs internally (see MILModel.predict) purely to surface three diagnostic
flags (`price_band_monotonicity_violated`/`boundary_hit`/`out_of_distribution`).
The production `recommended_price_*` fields are instead bundled in from the
separately-trained, validated m1_quote_price champion at the API layer (see
`_predict_mil_bundle` in services/configurator/app.py and its port in
maxflow/backend/app/src/service/prediction_service.py) — the same pattern
m1_quote_line_win already uses.

Certified-monotonic price branch (fixes a real bug: earlier versions of this
network could report a HIGHER win probability at a higher price for the same
line, and the network's own diagnostic flags would fire true on almost every
prediction). The network is now split into two additive branches whose logits
are summed before the final sigmoid:
  - the UNCONSTRAINED branch (`W1/b1/W2/b2/W3/b3`, the original 3-layer MLP)
    sees every feature EXCEPT price/margin: quantity, discount, unit cost, and
    the one-hot categoricals. Raw `salesPrice` was removed from
    its inputs entirely — with `margin_ratio` fed only to the branch below,
    leaving `salesPrice` here would let the network smuggle an unconstrained
    price signal straight back in through the back door.
  - the MONOTONIC branch (`mono_W1/mono_b1/mono_W2/mono_b2`) sees exactly one
    input: `-margin_ratio` (negated scaled margin, i.e. higher price -> lower
    input). Its two weight matrices are projected to be non-negative after
    every optimizer step (`np.clip(..., 0, None)` in `train_mlp`), and its
    only activation is ReLU — a non-negative-weighted sum of non-negative,
    non-decreasing (ReLU) functions of a non-decreasing input is itself
    non-decreasing (standard "certified monotonic network" construction via
    weight-sign constraints + projected gradient descent). So this branch's
    contribution to the logit can only fall (or stay flat) as price rises,
    for ANY weights the optimizer could possibly reach.
  - Because the total logit is `Z_u + Z_mono` (a plain sum) and sigmoid is
    monotonic increasing, `win_probability` is now non-increasing in price BY
    CONSTRUCTION, for every prediction — not just inside `recommend_price_band`'s
    internal sweep, but for any arbitrary `salesPrice` a caller submits to
    `/predict`. `price_band_monotonicity_violated` should now read False on
    essentially every prediction; if it ever fires True, that's a real
    regression in the weight-projection step, not an expected/known gap.
    Verified against 300 simulated training steps + a finite-difference
    gradient check on both branches (`tests/unit/test_m1_mil_guardrails.py`).

Known scope boundary: only `salesPrice` (via `margin_ratio`) is covered by this
guarantee. `discountPercent` is a second, independent price lever the raw data
carries — it is NOT included in the monotonic branch and could still, in
principle, produce a counter-intuitive discount response. Extending the
monotonic branch to cover discount too is a natural next step if that becomes
a concern, but was out of scope for this fix (the reported issue was
specifically about `salesPrice`)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from maxxflow_core.errors import get_logger
from maxxflow_mlops.naming import registered_model_name
from maxxflow_mlops.promotion import (expected_calibration_error, gate_metrics,
                                       should_promote)
from maxxflow_mlops.registry import MLflowRegistry
from m1_quote.csv_common import RunLogger, confidence_level

log = get_logger("m1_quote.csv_mil")
MODULE = "m1_quote_mil"
LABEL = "won"
GROUP = "quotationID"

# --- feature engineering ------------------------------------------------------
# NOTE: "margin_ratio" MUST stay first — _bag_forward_combined slices column 0
# off X as the sole input to the monotonic price branch (see module docstring).
# Raw "salesPrice" is deliberately NOT in this list (removed as part of the
# certified-monotonic-price fix): margin_ratio already carries the price signal,
# and letting raw salesPrice back in through the unconstrained branch would
# defeat the monotonicity guarantee.
SCALE_COLS = ["margin_ratio", "log_quantity", "discountPercent", "unitPrice"]
CATEG_COLS = ["industry", "region", "salesRepID"]
TOP_N_CATEGORIES = 8

# --- network / training hyperparameters (match the notebook's, minus torch) --
HIDDEN_DIMS = (16, 8)          # smaller than the notebook's (64,32,16): our
                                # dataset is a few thousand rows, not the scale
                                # that architecture was sized for
MONO_HIDDEN_DIM = 8            # width of the certified-monotonic price branch
                                # (single input -> MONO_HIDDEN_DIM -> 1)
DROPOUT_P = 0.2
EPOCHS = 1000
EARLY_STOP_PATIENCE = 75       # stop after this many consecutive epochs with no Val
                                # AUC improvement; only active when val_bags is passed
LEARNING_RATE = 1e-3
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1e-8
NOISY_OR_EPS = 1e-4             # clamp instance/bag probabilities away from 0/1
NOISY_OR_GRAD_CLIP = 10.0       # the 1/(1-yhat) term in the gradient explodes when
                                 # yhat sits near the clip boundary and disagrees with
                                 # the label — a known Noisy-OR MIL instability; clip
                                 # dZ3 at the source rather than let it cascade into NaNs
GRAD_NORM_CLIP = 5.0             # global gradient-norm clip (standard fix for weight
                                 # drift/explosion over many steps — the dZ3 clip alone
                                 # bounds one step's gradient but not cumulative drift)
N_MC_SAMPLES = 30               # MC-Dropout forward passes at serving time
MC_DROPOUT_SEED = 7             # fixed so identical inputs give identical output

# --- price-band optimizer (ported ~verbatim from the notebook; pure numpy) ---
PRICE_BAND_MIN_MARGIN = 0.0
PRICE_BAND_MAX_MARGIN = 0.35
PRICE_BAND_STEPS = 50

# --- "deal breaker" flag ------------------------------------------------------
DEAL_BREAKER_MARGIN_DELTA = 10.0   # percentage points above the OTHER lines' avg margin


# =============================================================================
# Feature engineering: derive -> scale -> one-hot (dense numpy matrix, no
# LightGBM-style native categoricals — a plain MLP needs numeric input only).
# =============================================================================
def _derive(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sales = pd.to_numeric(out["salesPrice"], errors="coerce")
    unit = pd.to_numeric(out["unitPrice"], errors="coerce")
    out["margin_ratio"] = np.where(sales > 0, (sales - unit) / sales, 0.0)
    out["log_quantity"] = np.log1p(pd.to_numeric(out["quantity"], errors="coerce").clip(lower=0).fillna(0.0))
    for c in ["unitPrice", "salesPrice", "discountPercent"]:
        out[c] = pd.to_numeric(out.get(c), errors="coerce").fillna(0.0)
    return out


def _collapse(series: pd.Series, top: list[str]) -> pd.Series:
    s = series.astype(str)
    return s.where(s.isin(top), other="Other")


def _encode_categoricals(df: pd.DataFrame, top_categories: dict) -> tuple[np.ndarray, list[str]]:
    parts, names = [], []
    for c in CATEG_COLS:
        top = top_categories.get(c, [])
        series = df[c] if c in df.columns else pd.Series(["Other"] * len(df), index=df.index)
        collapsed = _collapse(series, top)
        expected = top + ["Other"]
        dummies = pd.get_dummies(collapsed, prefix=c)
        dummies = dummies.reindex(columns=[f"{c}_{v}" for v in expected], fill_value=0.0)
        parts.append(dummies.to_numpy(dtype=np.float64))
        names.extend([f"{c}_{v}" for v in expected])
    X = np.hstack(parts) if parts else np.zeros((len(df), 0))
    return X, names


def transform_features(df: pd.DataFrame, scaler: StandardScaler, top_categories: dict,
                       already_derived: bool = False) -> tuple[np.ndarray, list[str]]:
    d = df if already_derived else _derive(df)
    scaled = scaler.transform(d[SCALE_COLS])
    cat_X, cat_names = _encode_categoricals(d, top_categories)
    X = np.hstack([scaled, cat_X]).astype(np.float64)
    return X, list(SCALE_COLS) + cat_names


def fit_feature_prep(df: pd.DataFrame):
    """Fits the scaler + top-category vocab on `df` (call with the TRAIN split
    only) — an explicit improvement over the notebook, which fits on the whole
    filtered dataset. Also captures the observed margin_ratio range: the price-band
    sweep in `recommend_price_band` later probes prices the network never trained
    on, and this range is what lets it detect that."""
    d = _derive(df)
    scaler = StandardScaler().fit(d[SCALE_COLS])
    top_categories = {c: d[c].astype(str).value_counts().nlargest(TOP_N_CATEGORIES).index.tolist()
                      for c in CATEG_COLS if c in d.columns}
    X, names = transform_features(d, scaler, top_categories, already_derived=True)
    margin_bounds = (float(d["margin_ratio"].min()), float(d["margin_ratio"].max()))
    return X, names, scaler, top_categories, margin_bounds


# =============================================================================
# Hand-written MLP: forward/backward pass + Adam. Fixed 2-hidden-layer
# architecture (input -> H1 -> H2 -> 1), ReLU + inverted dropout between
# layers, sigmoid output. No LayerNorm (the notebook's, needed there for
# batch-size-1 bags under BatchNorm's constraints; not needed with LayerNorm
# omitted entirely here — a deliberate scope cut, see the module docstring).
# =============================================================================
def _init_weights(input_dim: int, hidden_dims: tuple[int, int], rng: np.random.RandomState,
                  mono_hidden_dim: int = 0) -> dict:
    """Inits the unconstrained U-branch (`W1..b3`, unchanged from the original
    single-network design) and, if `mono_hidden_dim > 0`, ALSO the
    certified-monotonic price branch (`mono_W1/mono_b1/mono_W2/mono_b2`).

    `mono_hidden_dim` defaults to 0 (branch omitted, zero extra `rng` draws)
    deliberately, NOT to MONO_HIDDEN_DIM: existing/plain callers that pass only
    (input_dim, hidden_dims, rng) — e.g. the finite-difference tests that
    exercise the generic U-branch primitive on arbitrary dims — often reuse
    the same `rng` afterwards to draw synthetic data, and any extra draws
    consumed here would silently change that downstream data, not just add
    unused dict keys. Real training explicitly passes
    `mono_hidden_dim=MONO_HIDDEN_DIM` (see `train_mlp`)."""
    dims = [input_dim, hidden_dims[0], hidden_dims[1], 1]
    weights = {}
    for i in range(3):
        fan_in, fan_out = dims[i], dims[i + 1]
        std = np.sqrt(2.0 / fan_in)
        weights[f"W{i + 1}"] = rng.normal(0.0, std, size=(fan_in, fan_out))
        weights[f"b{i + 1}"] = np.zeros(fan_out)
    if mono_hidden_dim <= 0:
        return weights

    # Monotonic branch: single input (-margin_ratio) -> mono_hidden_dim -> 1.
    # Weights initialised NON-NEGATIVE (abs of a normal draw) so the network
    # starts inside the feasible region on step 0, before the first projection
    # in train_mlp even runs.
    std1 = np.sqrt(2.0 / 1)
    weights["mono_W1"] = np.abs(rng.normal(0.0, std1, size=(1, mono_hidden_dim)))
    weights["mono_b1"] = np.zeros(mono_hidden_dim)
    std2 = np.sqrt(2.0 / mono_hidden_dim)
    weights["mono_W2"] = np.abs(rng.normal(0.0, std2, size=(mono_hidden_dim, 1)))
    weights["mono_b2"] = np.zeros(1)
    return weights


def _dropout_mask(shape, p: float, rng: np.random.RandomState | None) -> np.ndarray:
    if p <= 0 or rng is None:
        return np.ones(shape)
    keep = (rng.random(shape) > p).astype(np.float64)
    return keep / (1.0 - p)


def _bag_forward(X: np.ndarray, weights: dict, dropout_p: float = 0.0,
                 rng: np.random.RandomState | None = None) -> tuple[np.ndarray, dict]:
    """The plain, UNCONSTRAINED 3-layer MLP — unchanged math from before this
    fix. Used two ways: (a) directly, by the finite-difference gradient tests
    (which don't care about the MIL/price-branch semantics, just correctness
    of this generic building block); (b) as the U-branch inside
    `_bag_forward_combined`, fed only the non-price feature columns."""
    Z1 = X @ weights["W1"] + weights["b1"]
    A1_raw = np.maximum(Z1, 0.0)
    M1 = _dropout_mask(A1_raw.shape, dropout_p, rng)
    A1 = A1_raw * M1

    Z2 = A1 @ weights["W2"] + weights["b2"]
    A2_raw = np.maximum(Z2, 0.0)
    M2 = _dropout_mask(A2_raw.shape, dropout_p, rng)
    A2 = A2_raw * M2

    Z3 = A2 @ weights["W3"] + weights["b3"]
    P = 1.0 / (1.0 + np.exp(-Z3))
    P = P.ravel()

    cache = {"X": X, "Z1": Z1, "A1": A1, "M1": M1, "Z2": Z2, "A2": A2, "M2": M2, "Z3": Z3}
    return P, cache


def _bag_backward(cache: dict, dZ3: np.ndarray, weights: dict) -> dict:
    X, Z1, A1, M1, Z2, A2, M2 = (cache["X"], cache["Z1"], cache["A1"], cache["M1"],
                                 cache["Z2"], cache["A2"], cache["M2"])
    dZ3 = dZ3.reshape(-1, 1)
    dW3 = A2.T @ dZ3
    db3 = dZ3.sum(axis=0)
    dA2 = (dZ3 @ weights["W3"].T) * M2
    dZ2 = dA2 * (Z2 > 0).astype(np.float64)

    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0)
    dA1 = (dZ2 @ weights["W2"].T) * M1
    dZ1 = dA1 * (Z1 > 0).astype(np.float64)

    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def _mono_forward(x: np.ndarray, weights: dict, dropout_p: float = 0.0,
                  rng: np.random.RandomState | None = None) -> tuple[np.ndarray, dict]:
    """The certified-monotonic price branch: 1 input -> mono_hidden_dim -> 1,
    ReLU, NO sigmoid (this returns a raw logit contribution to be summed with
    the U-branch's logit before the shared sigmoid — see `_bag_forward_combined`).
    `x` is expected to already be `-margin_ratio` (negated), so this branch only
    needs to be non-decreasing in its own input to make the overall network
    non-increasing in price. Dropout uses the same inverted-mask trick as
    `_bag_forward`: multiplying a non-negative, non-decreasing function by a
    non-negative mask value preserves non-decreasingness, so MC-Dropout
    averaging never breaks the guarantee."""
    Hpre = x @ weights["mono_W1"] + weights["mono_b1"]
    Hraw = np.maximum(Hpre, 0.0)
    Mm = _dropout_mask(Hraw.shape, dropout_p, rng)
    H = Hraw * Mm
    Z = H @ weights["mono_W2"] + weights["mono_b2"]
    cache = {"x": x, "Hpre": Hpre, "H": H, "Mm": Mm}
    return Z.ravel(), cache


def _mono_backward(cache: dict, dZ: np.ndarray, weights: dict) -> dict:
    x, Hpre, H, Mm = cache["x"], cache["Hpre"], cache["H"], cache["Mm"]
    dZ = dZ.reshape(-1, 1)
    dW2 = H.T @ dZ
    db2 = dZ.sum(axis=0)
    dH = (dZ @ weights["mono_W2"].T) * Mm
    dHpre = dH * (Hpre > 0).astype(np.float64)
    dW1 = x.T @ dHpre
    db1 = dHpre.sum(axis=0)
    return {"mono_W1": dW1, "mono_b1": db1, "mono_W2": dW2, "mono_b2": db2}


def _project_mono_weights(weights: dict) -> None:
    """The non-negativity constraint that makes the monotonic branch monotonic
    is enforced by projection (clip-after-step), not by construction of the
    optimizer update itself — call this after every Adam step during training.
    In-place; cheap (two small clips)."""
    weights["mono_W1"] = np.clip(weights["mono_W1"], 0.0, None)
    weights["mono_W2"] = np.clip(weights["mono_W2"], 0.0, None)


def _bag_forward_combined(X: np.ndarray, weights: dict, dropout_p: float = 0.0,
                          rng: np.random.RandomState | None = None) -> tuple[np.ndarray, dict]:
    """The real per-instance forward pass used everywhere in this module
    (training, MC-Dropout serving, and the price-band sweep): column 0 of X
    (scaled `margin_ratio`) is routed to the monotonic branch as `-margin_ratio`;
    every other column goes to the unconstrained U-branch. Logits are summed
    before the shared sigmoid, so `win_probability` is non-increasing in price
    by construction — see the module docstring for the full argument."""
    x_mono = -X[:, 0:1]
    X_u = X[:, 1:]
    _, u_cache = _bag_forward(X_u, weights, dropout_p, rng)
    Zu = u_cache["Z3"].ravel()
    z_mono, mono_cache = _mono_forward(x_mono, weights, dropout_p, rng)
    Z_total = Zu + z_mono
    P = 1.0 / (1.0 + np.exp(-Z_total))
    cache = {"u_cache": u_cache, "mono_cache": mono_cache}
    return P, cache


def _bag_backward_combined(cache: dict, dZ3: np.ndarray, weights: dict) -> dict:
    """dZ_total/dZu == dZ_total/dz_mono == 1 (the two branches are combined by
    a plain sum before the sigmoid), so the same upstream gradient `dZ3` feeds
    both branches' backward pass unchanged."""
    grads_u = _bag_backward(cache["u_cache"], dZ3, weights)
    grads_mono = _mono_backward(cache["mono_cache"], dZ3, weights)
    return {**grads_u, **grads_mono}


def noisy_or_grad(P: np.ndarray, y: float, eps: float = NOISY_OR_EPS) -> tuple[np.ndarray, float]:
    """Returns (dL/dz3 per instance, the bag's clipped Noisy-OR probability).
    See the module docstring for the derivation."""
    p = np.clip(P, eps, 1 - eps)
    yhat = float(np.clip(np.prod(p), eps, 1 - eps))
    dZ3 = (yhat - y) * (1 - p) / (1 - yhat)
    dZ3 = np.clip(dZ3, -NOISY_OR_GRAD_CLIP, NOISY_OR_GRAD_CLIP)
    return dZ3, yhat


class _Adam:
    def __init__(self, params: dict, lr: float = LEARNING_RATE):
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = ADAM_BETA1 * self.m[k] + (1 - ADAM_BETA1) * g
            self.v[k] = ADAM_BETA2 * self.v[k] + (1 - ADAM_BETA2) * (g * g)
            m_hat = self.m[k] / (1 - ADAM_BETA1 ** self.t)
            v_hat = self.v[k] / (1 - ADAM_BETA2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + ADAM_EPS)


def _clip_grad_norm(grads: dict, max_norm: float = GRAD_NORM_CLIP) -> dict:
    total_sq = sum(float(np.sum(g * g)) for g in grads.values())
    norm = np.sqrt(total_sq)
    if norm > max_norm and norm > 0:
        scale = max_norm / norm
        return {k: g * scale for k, g in grads.items()}
    return grads


def _make_bags(X: np.ndarray, df: pd.DataFrame, label_col: str = LABEL, group_col: str = GROUP):
    """df must have a clean 0..n-1 RangeIndex aligned with X's rows."""
    bags_X, bags_y, ids = [], [], []
    for qid, sub in df.groupby(group_col, sort=False):
        pos = sub.index.to_numpy()
        bags_X.append(X[pos])
        bags_y.append(float(sub[label_col].iloc[0]))
        ids.append(qid)
    return bags_X, bags_y, ids


def _bag_yhat(X_bag: np.ndarray, weights: dict, eps: float = NOISY_OR_EPS) -> float:
    P, _ = _bag_forward_combined(X_bag, weights, dropout_p=0.0)
    p = np.clip(P, eps, 1 - eps)
    return float(np.clip(np.prod(p), eps, 1 - eps))


def _evaluate_auc(weights: dict, bags: list[tuple[np.ndarray, float]]) -> float:
    yhats = [_bag_yhat(X, weights) for X, _ in bags]
    ys = [y for _, y in bags]
    if len(set(ys)) < 2:
        return float("nan")
    return float(roc_auc_score(ys, yhats))


def train_mlp(bags_X: list[np.ndarray], bags_y: list[float], input_dim: int, *,
             epochs: int = EPOCHS, lr: float = LEARNING_RATE, dropout_p: float = DROPOUT_P,
             seed: int = 7, logger: RunLogger | None = None,
             val_bags: list[tuple[np.ndarray, float]] | None = None,
             patience: int = EARLY_STOP_PATIENCE) -> tuple[dict, dict]:
    """`input_dim` is the U-BRANCH's input width, i.e. total engineered feature
    columns minus 1 (column 0, `margin_ratio`, goes to the monotonic branch
    instead — see `train()`'s call site and the module docstring).

    Early stopping only activates when `val_bags` is given (there's no other
    signal to stop on): the best-Val-AUC epoch's weights are snapshotted as
    they're found, and once `patience` epochs pass with no further improvement,
    training stops and THOSE weights are returned — not the last epoch's,
    which may already have overfit past the best point."""
    rng = np.random.RandomState(seed)
    weights = _init_weights(input_dim, HIDDEN_DIMS, rng, mono_hidden_dim=MONO_HIDDEN_DIM)
    opt = _Adam(weights, lr=lr)
    n_bags = len(bags_X)
    order = np.arange(n_bags)
    history = {"train_loss": [], "val_auc": []}
    best_val_auc = -np.inf
    best_weights = None
    epochs_no_improve = 0

    # BLAS matmul on some backends (e.g. Apple Accelerate) raises spurious
    # divide-by-zero/overflow/invalid RuntimeWarnings as an internal side
    # effect even when every value involved is finite (verified: cache/grad
    # values are always finite here) — same "ignore, don't chase phantom FP
    # flags" treatment the LightGBM models in this codebase already apply
    # around their own fit() calls.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for epoch in range(1, epochs + 1):
            rng.shuffle(order)
            running = 0.0
            for idx in order:
                X_bag, y = bags_X[idx], bags_y[idx]
                P, cache = _bag_forward_combined(X_bag, weights, dropout_p=dropout_p, rng=rng)
                dZ3, yhat = noisy_or_grad(P, y)
                running += -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
                grads = _bag_backward_combined(cache, dZ3, weights)
                grads = _clip_grad_norm(grads)
                opt.step(weights, grads)
                # Projected gradient step: this single line is what actually
                # enforces the monotonic-price guarantee. Adam's update has no
                # notion of the non-negativity constraint on mono_W1/mono_W2, so
                # it can (and does) push them negative; clipping back to >=0
                # immediately after every step is what keeps the branch's
                # "non-decreasing in -margin_ratio" property true throughout
                # training, not just at initialisation.
                _project_mono_weights(weights)
            train_loss = running / n_bags
            history["train_loss"].append(train_loss)

            should_log = epoch == 1 or epoch % 10 == 0 or epoch == epochs
            if val_bags is not None:
                val_auc = _evaluate_auc(weights, val_bags)
                history["val_auc"].append(val_auc)
                if logger and should_log:
                    logger.log(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.3f}")

                # NaN (single-class val split) carries no improvement signal —
                # skip it rather than let it count against patience.
                if not np.isnan(val_auc):
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        best_weights = {k: v.copy() for k, v in weights.items()}
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= patience:
                            if logger:
                                logger.log(f"Early stopping at epoch {epoch}/{epochs} "
                                          f"(no Val AUC improvement in {patience} epochs, "
                                          f"best={best_val_auc:.3f})")
                            break
            elif logger and should_log:
                logger.log(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f}")

    if best_weights is not None:
        weights = best_weights
    return weights, history


def mc_dropout_predict(X: np.ndarray, weights: dict, *, T: int = N_MC_SAMPLES,
                       dropout_p: float = DROPOUT_P, seed: int = MC_DROPOUT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """T stochastic forward passes with dropout kept ACTIVE (unlike normal
    serving) — the mean is the reported win_probability (matches the
    notebook), the spread converts to a [0,1] confidence score. Seeded so
    identical inputs give identical (reproducible) output."""
    rng = np.random.RandomState(seed)
    samples = np.zeros((T, X.shape[0]))
    for t in range(T):
        P, _ = _bag_forward_combined(X, weights, dropout_p=dropout_p, rng=rng)
        samples[t] = P
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    confidence = np.clip(1.0 - 2.0 * std, 0.0, 1.0)
    return mean, confidence


def recommend_price_band(weights: dict, scaler: StandardScaler, top_categories: dict,
                         row: dict, cost_price: float, quantity: float, *,
                         min_margin: float = PRICE_BAND_MIN_MARGIN, max_margin: float = PRICE_BAND_MAX_MARGIN,
                         steps: int = PRICE_BAND_STEPS,
                         margin_bounds: tuple[float, float] | None = None) -> dict:
    """Sweeps candidate sales prices across a margin band, holding every other
    feature fixed, and returns the price that maximises expected dollar
    margin = P(win|price) * (price-cost) * quantity — this model's OWN price
    recommendation from its OWN win-probability function (unlike
    m1_quote_line_win, which bundles the separately-trained m1_quote_price
    model). `low`/`high` are the swept range's bounds, not a statistical
    quantile band like the other models report.

    Three cheap guardrails on top of the raw sweep:
    - `monotonicity_violated`: since the certified-monotonic price fix, the
      network's own win-probability-vs-price curve is non-increasing BY
      CONSTRUCTION (see module docstring), so this should read False on
      essentially every call. It's kept as a live regression check — a
      cumulative-min (isotonic) projection still runs underneath so the
      optimizer can never be fooled even in the (now unexpected) case this
      fires — and the flag reports whether that fallback correction actually
      had to kick in. True here now means "investigate the weight-projection
      step", not "known model limitation".
    - `boundary_hit`: the chosen price sits at either end of the swept grid,
      i.e. the objective was still improving when the sweep ran out — the true
      optimum may lie outside [min_margin, max_margin] entirely. Still a live,
      expected guardrail: a monotonically falling win-probability curve doesn't
      make expected-margin unimodal, since margin itself keeps rising with price.
    - `out_of_distribution`: the chosen price's margin falls outside the
      margin_ratio range this model was actually trained on, so the network is
      extrapolating rather than interpolating at that point.
    """
    if cost_price <= 0:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "monotonicity_violated": False,
                "boundary_hit": False, "out_of_distribution": False}
    margins = np.linspace(min_margin, max_margin, steps)
    prices = cost_price / (1 - margins)

    win_probs = np.empty(steps)
    for i, price in enumerate(prices):
        r = dict(row)
        r["salesPrice"] = float(price)
        X, _ = transform_features(pd.DataFrame([r]), scaler, top_categories)
        p, _ = _bag_forward_combined(X, weights, dropout_p=0.0)
        win_probs[i] = p[0]

    # Enforce "higher price can't raise win probability" via a cumulative-min
    # projection (isotonic in the price-ascending direction) before optimizing,
    # so a non-monotonic artifact in the raw network output can't be picked as
    # the "optimal" price.
    monotonic_probs = np.minimum.accumulate(win_probs)
    monotonicity_violated = bool(np.any(win_probs > monotonic_probs + 1e-9))

    unit_margins = prices - cost_price
    expected = monotonic_probs * unit_margins * quantity
    best_idx = int(np.argmax(expected))
    boundary_hit = best_idx in (0, steps - 1)

    best_margin = float(margins[best_idx])
    out_of_distribution = (margin_bounds is not None
                           and not (margin_bounds[0] <= best_margin <= margin_bounds[1]))

    return {"low": float(prices.min()), "mid": float(prices[best_idx]), "high": float(prices.max()),
            "monotonicity_violated": monotonicity_violated, "boundary_hit": boundary_hit,
            "out_of_distribution": out_of_distribution}


def _deal_breaker_flags(margins: list[float]) -> list[bool]:
    n = len(margins)
    flags = []
    for i in range(n):
        others = [m for j, m in enumerate(margins) if j != i]
        if not others:
            flags.append(False)
            continue
        flags.append((margins[i] - (sum(others) / len(others))) > DEAL_BREAKER_MARGIN_DELTA)
    return flags


class MILModel(PythonModel):
    def __init__(self, weights: dict, scaler: StandardScaler, top_categories: dict,
                 product_support: dict, margin_bounds: tuple[float, float] | None = None,
                 provenance: str = "csv"):
        self.weights = weights
        self.scaler = scaler
        self.top_categories = top_categories
        self.product_support = product_support   # productID (str) -> historical row count
        self.margin_bounds = margin_bounds        # observed train-split margin_ratio (min, max)
        self.provenance = provenance

    def predict(self, context, model_input, params=None):
        df = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        df = df.reset_index(drop=True)

        group_key = "quotationID" if "quotationID" in df.columns and df["quotationID"].notna().any() else None
        groups = list(df.groupby(group_key, sort=False)) if group_key else [("_bag", df)]

        X, _ = transform_features(df, self.scaler, self.top_categories)
        win_mean, mc_conf = mc_dropout_predict(X, self.weights)

        margins = np.where(pd.to_numeric(df["salesPrice"], errors="coerce") > 0,
                           (pd.to_numeric(df["salesPrice"], errors="coerce")
                            - pd.to_numeric(df["unitPrice"], errors="coerce"))
                           / pd.to_numeric(df["salesPrice"], errors="coerce") * 100.0, 0.0)

        out_rows = []
        row_ptr = 0
        for _qid, sub in groups:
            n = len(sub)
            idx = np.arange(row_ptr, row_ptr + n)
            row_ptr += n

            p_clip = np.clip(win_mean[idx], NOISY_OR_EPS, 1 - NOISY_OR_EPS)
            quote_prob = float(np.clip(np.prod(p_clip), NOISY_OR_EPS, 1 - NOISY_OR_EPS))
            bag_margins = [float(margins[i]) for i in idx]
            deal_flags = _deal_breaker_flags(bag_margins)

            for j, i in enumerate(idx):
                row = sub.iloc[j]
                cost = float(pd.to_numeric(row.get("unitPrice"), errors="coerce") or 0.0)
                qty = float(pd.to_numeric(row.get("quantity"), errors="coerce") or 1.0)
                # recommend_price_band's own low/mid/high are still NOT served as the
                # production price recommendation, even though the network's price
                # response is now certified monotonic (see module docstring) —
                # m1_quote_price remains the validated, dedicated source for
                # recommended_price_low/mid/high; the API layer bundles its band in
                # (see _predict_mil_bundle). We still run the sweep here purely to
                # surface these three diagnostic flags.
                band = recommend_price_band(self.weights, self.scaler, self.top_categories,
                                            row.to_dict(), cost, qty,
                                            margin_bounds=getattr(self, "margin_bounds", None))
                product_id = str(row.get("productID", ""))
                n_comp = self.product_support.get(product_id, 0)
                out_rows.append({
                    "win_probability": round(float(win_mean[i]), 4),
                    "win_probability_pct": round(float(win_mean[i]) * 100, 1),
                    "mc_dropout_confidence": round(float(mc_conf[i]), 3),
                    "quote_win_probability": round(quote_prob, 4),
                    "price_band_monotonicity_violated": bool(band["monotonicity_violated"]),
                    "price_band_boundary_hit": bool(band["boundary_hit"]),
                    "price_band_out_of_distribution": bool(band["out_of_distribution"]),
                    "deal_breaker": bool(deal_flags[j]),
                    "n_comparable": int(n_comp),
                    "confidence": confidence_level(int(n_comp)),
                })
        return pd.DataFrame(out_rows)


_NUMERIC_EXAMPLE_COLS = ["quantity", "unitPrice", "salesPrice", "discountPercent"]
_CATEG_EXAMPLE_COLS = ["quotationID", "productID", "region", "industry", "salesRepID"]


def _example_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = _NUMERIC_EXAMPLE_COLS + _CATEG_EXAMPLE_COLS
    ex = df[cols].head(3).copy()
    for c in _NUMERIC_EXAMPLE_COLS:
        ex[c] = pd.to_numeric(ex[c], errors="coerce").astype(float)
    for c in _CATEG_EXAMPLE_COLS:
        ex[c] = ex[c].astype(str)
    return ex


def train(df_or_path, tenant: str, *, auto_hpo: bool = True, register: bool = True,
          source: str = "csv", logger: RunLogger | None = None) -> dict:
    logger = logger or RunLogger()
    df = pd.read_csv(df_or_path) if isinstance(df_or_path, str) else df_or_path.copy()

    required = ["quotationID", LABEL, "productID", "quantity", "unitPrice", "salesPrice"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"dataset missing required columns: {missing}")

    if "tenant" in df.columns:
        present = sorted(map(str, df["tenant"].dropna().unique()))
        logger.log(f"'tenant' column present {present} — ignored for modelling; training on "
                   f"all {len(df)} lines across {df['quotationID'].nunique()} quotes "
                   f"(tenant only names the model '{tenant}')")
    else:
        logger.log(f"Loaded {len(df)} lines across {df['quotationID'].nunique()} quotes "
                   f"(source={source}, model tenant={tenant})")

    df[LABEL] = pd.to_numeric(df[LABEL], errors="coerce")
    df = df.dropna(subset=[LABEL, "quotationID", "unitPrice", "salesPrice"]).reset_index(drop=True)
    df[LABEL] = df[LABEL].astype(int)
    if df[LABEL].nunique() < 2:
        raise ValueError(f"tenant {tenant}: only one class present — cannot train")

    product_support = df["productID"].astype(str).value_counts().to_dict()

    unique_ids = df["quotationID"].unique()
    qid_label = df.groupby("quotationID")[LABEL].first()
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=7,
                                           stratify=qid_label.loc[unique_ids].to_numpy())
    train_df = df[df["quotationID"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["quotationID"].isin(test_ids)].reset_index(drop=True)
    logger.log(f"Split (by quote, not by line — avoids leaking a quote's other lines across "
               f"the split): {train_df['quotationID'].nunique()} train / "
               f"{test_df['quotationID'].nunique()} holdout quotes "
               f"({len(train_df)} / {len(test_df)} lines)")

    Xtr, feature_names, scaler, top_categories, margin_bounds = fit_feature_prep(train_df)
    Xte, _ = transform_features(test_df, scaler, top_categories)
    logger.log(f"Engineered {len(feature_names)} features ({len(SCALE_COLS)} scaled numeric + "
               f"one-hot top-{TOP_N_CATEGORIES} categoricals + Other, fit on the train split "
               f"only); column 0 (margin_ratio) is routed to the certified-monotonic price "
               f"branch, the remaining {len(feature_names) - 1} go to the unconstrained branch")

    train_bags_X, train_bags_y, _ = _make_bags(Xtr, train_df)
    test_bags_X, test_bags_y, _ = _make_bags(Xte, test_df)
    val_bags = list(zip(test_bags_X, test_bags_y))

    # input_dim is the U-BRANCH's width: total engineered columns minus the one
    # (margin_ratio, column 0) carved off for the monotonic price branch.
    weights, _history = train_mlp(train_bags_X, train_bags_y, input_dim=Xtr.shape[1] - 1,
                                  epochs=EPOCHS, lr=LEARNING_RATE, dropout_p=DROPOUT_P,
                                  seed=7, logger=logger, val_bags=val_bags)

    yhats = [_bag_yhat(X, weights) for X in test_bags_X]
    ys = test_bags_y
    pred = [1 if p >= 0.5 else 0 for p in yhats]
    tn, fp, fn, tp = confusion_matrix(ys, pred, labels=[0, 1]).ravel()
    _ys = np.array(ys, dtype=float)
    # majority-class accuracy on this quote-level holdout, for the promotion floor
    _base_rate = float(max(_ys.mean(), 1 - _ys.mean())) if len(_ys) else 0.0
    metrics = {
        "accuracy": float(np.mean(np.array(pred) == np.array(ys))),
        "base_rate_accuracy": _base_rate,
        "accuracy_over_base_rate": float(np.mean(np.array(pred) == np.array(ys)) - _base_rate),
        "auc": float(roc_auc_score(ys, yhats)) if len(set(ys)) > 1 else 0.5,
        "pr_auc": float(average_precision_score(ys, yhats)) if len(set(ys)) > 1 else 0.5,
        "precision": float(precision_score(ys, pred, zero_division=0)),
        "recall": float(recall_score(ys, pred, zero_division=0)),
        "f1": float(f1_score(ys, pred, zero_division=0)),
        "brier": float(brier_score_loss(ys, yhats)),
        "ece": float(expected_calibration_error(np.array(ys, dtype=float), np.array(yhats))),
        "n_train_quotes": int(len(train_bags_y)), "n_test_quotes": int(len(test_bags_y)),
        "n_train": int(len(train_df)), "n_test": int(len(test_df)),
        "positive_rate": float(df[LABEL].mean()),
    }
    logger.log(f"Metrics (quote-level, held-out): accuracy={metrics['accuracy']:.3f} "
               f"AUC={metrics['auc']:.3f} Brier={metrics['brier']:.3f} ECE={metrics['ece']:.3f}")

    model = MILModel(weights, scaler, top_categories, product_support, margin_bounds, provenance=source)
    params = {"hidden_dims": list(HIDDEN_DIMS), "mono_hidden_dim": MONO_HIDDEN_DIM,
              "epochs": EPOCHS, "lr": LEARNING_RATE, "dropout_p": DROPOUT_P,
              "algo": "noisy-or-mil-numpy-certified-monotonic-price"}

    result = {"model_type": "classification", "metrics": metrics, "features": feature_names,
              "dropped_features": [], "source": source,
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
    ex = _example_frame(df)
    sig = infer_signature(ex, model.predict(None, ex))
    version = reg.log_and_register(model, name=name, params=params, metrics=metrics,
                                   tags=tags, signature=sig, input_example=ex)
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
        # MIL scores at QUOTE level, so the ECE/Brier denominator is the number of
        # held-out QUOTES, not the line count that metrics["n_test"] carries.
        gate_metrics(candidate_metrics, n_test=candidate_metrics.get("n_test_quotes")),
        None if champ is None else {"auc": champ["auc"], "brier": champ["brier"]}, force=force)
    if decision.promote:
        reg.promote(name=name, challenger_version=str(version))
    return {"published": decision.promote, "reasons": decision.reasons,
            # the single DECIDING check when refused. A caller showing reasons[0]
            # next to a refusal could otherwise quote a check that passed.
            "blocker": decision.blocker, "gate_checks": decision.checks,
            "gate_summary": decision.summary,
            "champion_before": champ, "candidate": candidate_metrics}
