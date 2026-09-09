"""M1 Smart Quote Optimiser — PER-PRODUCT WIN & PRICE model (US TA/SM 10.1.1 + 10.1.2).

This module owns the whole **AI Insights card** for one quotation line. A single
``predict()`` call returns all three sections the BRD specifies, so the panel is
driven by one model and the guardrails have exactly one implementation:

  1. **Win Probability**   — calibrated probability, its display band
     (Green >= 65% / Amber 40-64% / Red < 40%) and an honest uncertainty interval.
  2. **Recommended Price Band** — the price range that maximises EXPECTED MARGIN
     (win probability x margin), swept only across prices this model has evidence
     for, then clamped to +/-30% of the tenant base Sales Price.
  3. **Confidence Level**  — High/Low card state plus the BRD guardrail message.

Ground truth (won/lost) is only recorded per QUOTATION, never per line, so
training rows come from ``raw_ingest.build_line_frame`` /
``db_features.build_line_frame``, which BROADCAST the quote's label onto every
one of its lines. That is a deliberate modelling choice, not an oversight — but
it has two consequences this module has to handle explicitly, and both were
previously unhandled:

  * **Sibling lines are not independent.** Every line of a quote shares its
    label AND its header features (region, industry, leadTimeDays, the two
    win-rates). A row-wise split therefore puts siblings on both sides and the
    model can recognise a holdout quote from its header. All splitting here is
    GROUPED BY QUOTATION, and TIME-ORDERED when a quote date is available —
    production retrains monthly and predicts forward, so a forward-in-time
    holdout is the only honest estimate of what the panel will do next month.
  * **The displayed number is a frequency.** "85%" has to mean "roughly 85 of
    100 comparable quotes won", so calibration is fitted on its OWN fold and
    scored on a third, untouched holdout. Fitting isotonic on the holdout and
    then reporting Brier/ECE on that same holdout — which is what this module
    used to do, and what the promotion gate reads — flatters exactly the two
    metrics the gate depends on.

Business logic that lives here rather than in the frontend
----------------------------------------------------------
* **Price only ever moves the score one way.** ``price_ratio`` carries a
  LightGBM monotone constraint (-1), so raising the price can never raise the
  displayed win probability. Without it an unconstrained tree fits the
  endogeneity in this data — reps discount hardest on deals already in trouble —
  and the panel would tell a rep that discounting *lowers* their chances.
* **The band is a decision, not a description.** Fitting quantiles to winning
  prices only describes what survived. Here the band is the set of prices within
  ``EV_BAND_TOL`` of the expected-margin optimum. When the price response is
  flat — which is what the M1 price-sensitivity analysis found in this data —
  the margin term dominates and the band correctly moves UP, i.e. it recommends
  discounting less.
* **Comparables are counted in QUOTATIONS, not lines** (BRD: "fewer than 5
  comparable historical quotations"). One quote carrying the same product on
  five lines is one comparable, not five.
* **Cold start falls back down a hierarchy** productID -> product_type ->
  none, because with ~65 decided lines per product the five-comparable guardrail
  would otherwise blank most of the catalogue on day one.

Adaptive by design: the BRD names seven training data elements and only some are
present in today's gold frames. Anything absent is skipped and reported in the
run log rather than synthesised, so a retrain never silently invents a feature.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from mlflow.models import ModelSignature, infer_signature
from mlflow.pyfunc import PythonModel
from mlflow.types import ColSpec, DataType, Schema
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from maxxflow_core.errors import get_logger
from maxxflow_core.money import clamp_price, is_price_clamped
from maxxflow_mlops.naming import registered_model_name
from maxxflow_mlops.promotion import (expected_calibration_error, gate_metrics,
                                       should_promote)
from maxxflow_mlops.registry import MLflowRegistry
from maxxflow_features.cleaning import clean_frame
from m1_quote.csv_common import RunLogger, auto_tune_classifier, confidence_level
from m1_quote.model import LOW_CONF_HIGH, LOW_CONF_LOW, MIN_COMPARABLE

log = get_logger("m1_quote.csv_line_win")
MODULE = "m1_quote_line_win"
LABEL = "won"
GROUP = "productID"          # what a "comparable" is counted within
QUOTE_COL = "quotationID"    # the true independence unit — splits group on this
DATE_COL = "quoteDate"       # when present, the holdout is forward in time
PRICE_FEATURE = "price_ratio"   # effective sales price / unit cost
COST_FEATURE = "unitPrice"      # unit COST in this schema (see raw_ingest)
# price_ratio is measured against LIST price when the export has one, else against
# cost. The band sweep has to convert a ratio back into money with the same basis,
# or it would recommend a price on a scale the model never used.
BASIS_FEATURE = "list_price"

# Core feature set — unchanged, and every one of these is produced by
# raw_ingest.build_line_frame today.
LINE_NUMERIC = ["quantity", "unitPrice", "price_ratio", "leadTimeDays",
                "contact_win_rate", "salesrep_win_rate"]
LINE_CATEG = ["productID", "region", "industry"]

# The remaining BRD "Training Data Elements", plus the extras an export may
# carry. Picked up the moment ingest supplies them; named in the run log when it
# cannot, so a retrain never quietly trains on less than it could.
BRD_NUMERIC = ["quote_total", "days_to_expiry"]
BRD_CATEG = ["product_type", "payment_terms"]

# As-of history and deal-shape features from raw_ingest.build_line_frame. Every
# one is computed strictly BEFORE its quote's date, so none can see its own
# outcome. The relative-price pair matters most: absolute margin over cost is not
# comparable across a catalogue whose unit costs span orders of magnitude, so
# "dearer than we usually sell this product for" carries the signal that
# price_ratio alone cannot.
HISTORY_NUMERIC = ["price_vs_product", "price_vs_customer", "product_win_rate",
                   "value_vs_customer", "leadtime_vs_product", "line_share",
                   "quote_month", "customer_prior_quotes", "customer_recency_days"]

# Derived inside this module (train AND serve) so both the CSV and DB frames get
# it without either ingest path having to know — but ONLY when the data earns it,
# see _below_cost_helps().
BELOW_COST = "below_cost"
DERIVED_NUMERIC = [BELOW_COST]
BELOW_COST_MIN_ROWS = 200    # too few loss-making lines to measure a dip
BELOW_COST_MIN_DIP = 0.02    # win rate must actually fall below cost, not just wobble

# BRD element -> the column that carries it, for the coverage line in the log.
BRD_ELEMENTS = {
    "Total (inc. GST)": "quote_total",
    "Sales Price & Unit Cost": "price_ratio",
    "Product Type": "product_type",
    "Contact": "contact_win_rate",
    "Salesperson": "salesrep_win_rate",
    "Expiration Date": "days_to_expiry",
    "Quotation Status": LABEL,
}

FINALIZED = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20}
# Early stopping watches the CALIBRATION fold (never Xte, so the promotion-gate
# metrics stay honest) and picks the boosting round count for us — the tuned
# n_estimators above becomes a generous cap instead of a fixed target, so a
# large upload can't overfit past the point the calibration fold shows the
# model has stopped improving. AUC on a small fold is too noisy a stopping
# signal (a few dozen rows can swing early-stop to a handful of trees and
# underfit badly), so EARLY_STOP_MIN_ROWS gates it off small test fixtures and
# small real uploads alike — those keep the fixed, tuned n_estimators exactly
# as before. Real CSV exports run tens of thousands of rows, so this only
# changes behaviour where there is enough signal for it to be trustworthy.
EARLY_STOP_ROUNDS = 50
EARLY_STOP_MAX_ROUNDS = 2000
EARLY_STOP_MIN_ROWS = 300

# --- split -----------------------------------------------------------------
# Three folds, never two: fit / calibrate / score. Grouped by quotation always;
# ordered by date when there is one.
TRAIN_FRAC, CALIB_FRAC = 0.60, 0.20      # holdout takes the remaining 0.20
MIN_FOLD_ROWS = 20                        # below this a fold cannot support isotonic
SPLIT_SEEDS = tuple(range(7, 27))         # retried until every fold has both classes

# --- uncertainty -----------------------------------------------------------
N_BOOTSTRAP = 15          # training-resample variance (input-dependent width)
CI_LOW_PCT, CI_HIGH_PCT = 5, 95
WILSON_Z = 1.6449         # 90% two-sided, for calibration sampling error
CALIB_BINS = 10

# --- Recommended Price Band ------------------------------------------------
SWEEP_POINTS = 41         # resolution of the price sweep

# --- what the band optimises ------------------------------------------------
# WIN PROBABILITY, subject to never quoting below cost. That is the business
# requirement: "the price range that maximises the chance of winning".
#
# The constraint is not optional. Win probability is monotone non-increasing in
# price, so "maximise it" unconstrained has a degenerate answer — quote zero and
# win everything. `unitPrice` is real cost in this schema (the tenant's own rule),
# so the floor is cost: the cheapest price we will ever recommend is the one that
# breaks even, and that price also carries the HIGHEST win probability available.
#
# The band then runs UP from that floor for as long as the odds hold: every price
# inside it is within PROB_BAND_TOL of the best achievable probability. `mid` is
# the highest price still within half that tolerance — the same odds for more
# margin, which is free money rather than a trade-off.
#
# The expected-margin optimum is still computed and returned as `ev_price_*`, so
# the two rules can be compared, but it is NOT the recommendation.
PROB_BAND_TOL = 0.05      # band = prices whose win probability is within 5 points of the best
EV_BAND_TOL = 0.10        # secondary: prices within 10% of peak expected margin
OBS_LO_PCT, OBS_HI_PCT = 1.0, 99.0   # never sweep outside observed pricing

# --- the dead zone ---------------------------------------------------------
# Observed pricing (the 1/99 percentiles above) is NOT the same thing as pricing
# the model can score. A booster has no split below its lowest split point, so
# every price under it lands in the same leaf and returns the SAME probability —
# a flat region where the answer looks confident and carries no information.
# Measured on the demo champion: price_bounds started at ratio 0.696, but the
# response did not move until 0.77, and the recommended band sat at 0.57-0.64,
# i.e. entirely inside the flat region. The panel was recommending a price the
# model cannot score, then reporting a constant 89.9% next to it.
#
# So the sweep is restricted to the INFORMATIVE range, measured after fitting by
# asking the model where its own answer actually changes. A recommendation can
# then only ever land where there is evidence.
FLAT_TOL = 0.005          # a response move under half a point is no move at all
FLAT_PROBE_POINTS = 121   # resolution of the flatness probe
FLAT_PROBE_ROWS = 300     # rows sampled for the probe

# --- displayed probability floor/ceiling ------------------------------------
# Isotonic on a modest calibration fold saturates to exactly 0 and 1. A panel
# that reads "100%" is wrong on its face — the number is a calibrated frequency,
# and this codebase's own leakage gate treats a perfect score as a defect — so
# the served probability is bounded away from both ends. Note 0.99 still trips
# the BRD's >95% low-confidence guardrail, which is the intended outcome.
PROB_FLOOR, PROB_CEIL = 0.01, 0.99

# --- BRD display thresholds -------------------------------------------------
# Red < 40 and Amber 40-64 share their lower edge with the low-confidence floor,
# so LOW_CONF_LOW is reused rather than repeated as a second magic number.
PROB_BAND_HIGH = 0.65
PROB_BAND_MED = LOW_CONF_LOW

MSG_SUFFICIENT = "Prediction based on sufficient historical data"
MSG_FEW_COMPARABLE = "Flagged by guardrail. Few comparable quotes"
# The BRD attaches MSG_FEW_COMPARABLE to the <40%/>95% rule too, but an extreme
# score is not evidence of thin data — same amber card, accurate sentence.
MSG_SCORE_RANGE = "Flagged by guardrail. Score outside reliable range"
# A price outside anything the model was trained on cannot be scored honestly.
# The monotone constraint means every such line returns the SAME floor (or ceiling)
# probability, which looks like a confident answer and is not one — that is exactly
# how a train/serve basis mismatch hid itself: two different products, identical
# score. Say so on the card instead.
MSG_PRICE_RANGE = "Flagged by guardrail. Price outside the range this model has seen"


class _IdentityCalibrator:
    """Stand-in when a calibration fold is too small or single-class. Module-level
    (not a lambda/closure) so the model still pickles for the registry."""

    def predict(self, x):
        return np.asarray(x, dtype=float)


def _with_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add `below_cost`, the one thing the monotone constraint cannot express.

    The win/price relationship is sometimes a HUMP rather than a slope: above
    cost a higher price loses deals (the constraint captures that), but the very
    cheapest lines can also lose, because a line quoted under unit cost is a
    distress signal rather than a bargain. A monotone-decreasing curve cannot
    hold both shapes at once, so the below-cost half gets its own unconstrained
    binary feature and `price_ratio` keeps its guarantee over the range reps
    actually trade in.

    Always computed; whether it is USED is decided per dataset by
    _below_cost_helps(), because on an export with no distress dip it buys
    nothing and costs the clean full-sweep monotonicity."""
    if PRICE_FEATURE not in df.columns or BELOW_COST in df.columns:
        return df
    out = df.copy()
    out[BELOW_COST] = (pd.to_numeric(df[PRICE_FEATURE], errors="coerce") < 1.0).astype(float)
    return out


def _below_cost_helps(df: pd.DataFrame, y: pd.Series, logger: RunLogger) -> bool:
    """Is there actually a distress dip below cost in THIS export?

    Compares the win rate of loss-making lines against the cheapest quartile of
    profitable ones — its nearest neighbour in price, so the comparison is not
    confounded by the overall downward slope. Measured on the whole frame because
    it decides the feature set, not a parameter."""
    if PRICE_FEATURE not in df.columns:
        return False
    r = pd.to_numeric(df[PRICE_FEATURE], errors="coerce")
    below = r < 1.0
    if int(below.sum()) < BELOW_COST_MIN_ROWS or int((~below).sum()) < BELOW_COST_MIN_ROWS:
        logger.log(f"'{BELOW_COST}' skipped: only {int(below.sum())} loss-making lines — too few "
                   f"to tell a distress dip from noise (need {BELOW_COST_MIN_ROWS})")
        return False
    above = r[~below]
    near = (~below) & (r <= above.quantile(0.25))    # cheapest profitable lines
    wr_below, wr_near = float(y[below].mean()), float(y[near].mean())
    dip = wr_near - wr_below
    use = dip >= BELOW_COST_MIN_DIP
    logger.log(f"Below-cost check: {int(below.sum())} lines under unit cost win {wr_below:.1%} vs "
               f"{wr_near:.1%} for the cheapest profitable quartile (dip {dip:+.1%}) — "
               + (f"adding '{BELOW_COST}' so the price curve can bend at cost"
                  if use else f"no distress dip, '{BELOW_COST}' omitted and the price response "
                              f"stays monotone across the whole sweep"))
    return use


def _select_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    num = [c for c in LINE_NUMERIC + BRD_NUMERIC + HISTORY_NUMERIC + DERIVED_NUMERIC
           if c in df.columns]
    cat = [c for c in LINE_CATEG + BRD_CATEG if c in df.columns]
    return num, cat


def _monotone_constraints(features: list[str]) -> list[int]:
    """-1 on the price feature: the win probability may never RISE with price.

    This is the single most important line of business logic in the module. The
    M1 price-sensitivity work showed the raw relationship in this data runs the
    wrong way (deeper discounts close less often, because reps discount troubled
    deals), so an unconstrained booster will happily learn it and the panel will
    reward price rises. The constraint makes that unrepresentable, and it is
    also what makes the expected-margin sweep below well-posed."""
    return [-1 if f == PRICE_FEATURE else 0 for f in features]


def _build_signature(example: pd.DataFrame, output) -> ModelSignature:
    """Log the BRD extras as OPTIONAL inputs, not required ones.

    MLflow enforces the logged input schema before ``predict`` is ever called, so
    anything marked required is a hard serving contract. The core line columns
    genuinely are — without a price and a product there is nothing to score. The
    BRD extras (quote_total, product_type, payment_terms, days_to_expiry) are
    not: this model already degrades an absent feature to the TRAINING median or
    an unknown category, which is strictly better than refusing to answer.

    Getting this wrong is a live outage, not a modelling nicety — a champion
    trained on an export that HAD product_type would 500 every prediction from a
    caller that does not send it, which is exactly what happened the first time
    these features were added. Marking them optional keeps both callers working,
    and a caller that does send them still has them reach the model (MLflow drops
    input columns that are absent from the schema entirely, so leaving them out
    would silently disable the features instead)."""
    cats = set(LINE_CATEG + BRD_CATEG)
    core = set(LINE_NUMERIC + LINE_CATEG)   # everything else is optional at serve time
    outputs = infer_signature(example, output).outputs
    try:
        cols = [ColSpec(DataType.string if c in cats else DataType.double, c,
                        required=(c in core)) for c in example.columns]
        return ModelSignature(inputs=Schema(cols), outputs=outputs)
    except TypeError:      # mlflow < 2.10: no `required=` — fall back to all-required
        return infer_signature(example, output)


def _brd_coverage(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    have = [name for name, col in BRD_ELEMENTS.items() if col in df.columns]
    missing = [f"{name} [{col}]" for name, col in BRD_ELEMENTS.items() if col not in df.columns]
    return have, missing


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------
def _walk_groups(order: list, sizes: dict, n: int) -> tuple[list, list, list]:
    """Assign whole groups, in the given order, into train/calib/holdout by row count."""
    tr, ca, te = [], [], []
    seen = 0
    for g in order:
        if seen < TRAIN_FRAC * n:
            tr.append(g)
        elif seen < (TRAIN_FRAC + CALIB_FRAC) * n:
            ca.append(g)
        else:
            te.append(g)
        seen += sizes[g]
    return tr, ca, te


def _folds_usable(y: pd.Series, idx: tuple) -> bool:
    return all(len(i) >= MIN_FOLD_ROWS and y.iloc[i].nunique() >= 2 for i in idx)


def _three_way_split(df: pd.DataFrame, y: pd.Series,
                     logger: RunLogger) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """fit / calibrate / score, grouped by quotation and time-ordered when possible.

    Falls back in a documented order rather than silently: temporal-grouped ->
    random-grouped -> row-wise stratified (only when the frame carries no
    quotation id at all, e.g. a hand-made frame in a test)."""
    n = len(df)
    pos = np.arange(n)

    if QUOTE_COL in df.columns:
        groups = df[QUOTE_COL].astype(str).to_numpy()
        uniq = pd.unique(groups)
        sizes = pd.Series(groups).value_counts().to_dict()
        by_group = {g: pos[groups == g] for g in uniq}

        def build(order):
            tr, ca, te = _walk_groups(list(order), sizes, n)
            return tuple(np.concatenate([by_group[g] for g in part]) if part else np.array([], int)
                         for part in (tr, ca, te))

        if DATE_COL in df.columns:
            dates = pd.to_datetime(df[DATE_COL], errors="coerce")
            if dates.notna().any():
                order = (pd.DataFrame({"g": groups, "d": dates})
                         .groupby("g")["d"].min().sort_values().index.tolist())
                idx = build(order)
                if _folds_usable(y, idx):
                    logger.log(f"Split: grouped by {QUOTE_COL}, ORDERED BY {DATE_COL} — holdout is "
                               f"forward in time, matching the monthly retrain-and-predict-forward "
                               f"pattern ({len(uniq)} quotes)")
                    return (*idx, "temporal-grouped")
                logger.log(f"Split: temporal grouping left a fold single-class or under "
                           f"{MIN_FOLD_ROWS} rows — falling back to random grouping")

        for seed in SPLIT_SEEDS:
            order = pd.Series(uniq).sample(frac=1.0, random_state=seed).tolist()
            idx = build(order)
            if _folds_usable(y, idx):
                logger.log(f"Split: grouped by {QUOTE_COL} (random, seed={seed}) — sibling lines of "
                           f"a quote share a label and every header feature, so they never straddle "
                           f"folds ({len(uniq)} quotes)")
                return (*idx, "random-grouped")

    logger.log(f"Split: no usable '{QUOTE_COL}' grouping — falling back to a row-wise stratified "
               f"split. Metrics from this path are OPTIMISTIC when lines share a quotation.")
    tr, rest = train_test_split(pos, test_size=1 - TRAIN_FRAC, random_state=7, stratify=y)
    rel = CALIB_FRAC / (1 - TRAIN_FRAC)
    ca, te = train_test_split(rest, test_size=1 - rel, random_state=7, stratify=y.iloc[rest])
    return tr, ca, te, "stratified-rowwise"


# --------------------------------------------------------------------------
# uncertainty
# --------------------------------------------------------------------------
def _wilson(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval on an observed frequency — the sampling error in the
    calibration data behind a displayed percentage. Degrades to [0,1] on n=0
    rather than pretending to know anything."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def _calibration_bands(p_cal: np.ndarray, y_cal: np.ndarray,
                       bins: int = CALIB_BINS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin Wilson interval on the calibration fold's realised win rate.

    Answers the question the UI actually poses: of the comparable quotes that
    scored around here, what fraction really closed, and how well pinned down is
    that? A sparsely-populated region of the score range yields a wide band on
    its own, with no tuning."""
    n = len(p_cal)
    edges = np.unique(np.quantile(p_cal, np.linspace(0, 1, min(bins, max(2, n // 10)) + 1)))
    if len(edges) < 2:
        lo, hi = _wilson(int(y_cal.sum()), n)
        return np.array([0.0, 1.0]), np.array([lo]), np.array([hi])
    which = np.clip(np.searchsorted(edges, p_cal, side="right") - 1, 0, len(edges) - 2)
    los, his = [], []
    for b in range(len(edges) - 1):
        m = which == b
        lo, hi = _wilson(int(y_cal[m].sum()), int(m.sum()))
        los.append(lo)
        his.append(hi)
    return edges, np.array(los), np.array(his)


# --------------------------------------------------------------------------
# served model
# --------------------------------------------------------------------------
class LineWinModel(PythonModel):
    """Everything the AI Insights card needs, from one call.

    Every attribute added after the first release is read through ``getattr``:
    a champion pickled by an older build deserialises straight into ``__dict__``
    without running ``__init__``, and must keep predicting rather than raising."""

    def __init__(self, booster, calibrator, categories: dict, feature_columns: list[str],
                 product_support: dict, group_col: str | None, provenance="csv",
                 bootstrap_models: list | None = None, numeric_fill: dict | None = None,
                 calib_edges=None, calib_low=None, calib_high=None,
                 price_bounds: tuple | None = None, product_type_support: dict | None = None,
                 support_basis: str = "lines", price_basis: str = "unknown",
                 price_informative_bounds: tuple | None = None):
        self.booster = booster
        self.calibrator = calibrator
        self.categories = categories
        self.feature_columns = feature_columns
        self.product_support = product_support     # productID (str) -> comparable count
        self.group_col = group_col
        self.provenance = provenance
        self.bootstrap_models = bootstrap_models or []
        # Training medians, so a missing value means the same thing at serve time
        # as it did at fit time. Filling with 0.0 here (the previous behaviour)
        # scored a missing leadTimeDays as "due today".
        self.numeric_fill = numeric_fill or {}
        self.calib_edges = calib_edges
        self.calib_low = calib_low
        self.calib_high = calib_high
        self.price_bounds = price_bounds           # observed (min, max) price_ratio
        self.product_type_support = product_type_support or {}
        self.support_basis = support_basis         # "quotations" | "lines"
        # What price_ratio was measured AGAINST at training time. Recorded so a
        # caller can check it rather than assume, and so the mismatch that produced
        # identical scores for every line is detectable after the fact.
        self.price_basis = price_basis
        # Where the price response actually moves. Narrower than price_bounds
        # whenever the booster has no split near the edge of observed pricing.
        # Everything outside this is the dead zone: a constant dressed as a score.
        self.price_informative_bounds = price_informative_bounds

    # -- helpers ------------------------------------------------------------
    def _attr(self, name, default):
        v = getattr(self, name, None)
        return default if v is None else v

    def _coerce(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _with_derived(df)      # below_cost is derived, never supplied
        X = pd.DataFrame(index=df.index)
        fill = self._attr("numeric_fill", {})
        for c in self.feature_columns:
            if c in self.categories:
                cats = self.categories[c]
                src = (pd.Series(df[c], index=df.index) if c in df.columns
                       else pd.Series([None] * len(df), index=df.index)).astype(object)
                # A product/region never seen in training becomes missing rather
                # than an unknown level — LightGBM handles missing, and the
                # comparables guardrail is what actually flags it to the user.
                X[c] = pd.Categorical(src.where(src.isin(cats)), categories=cats)
            else:
                src = df[c] if c in df.columns else pd.Series(np.nan, index=df.index)
                X[c] = pd.to_numeric(src, errors="coerce").fillna(fill.get(c, 0.0))
        return X

    def _calibrated(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict_proba(X[self.feature_columns])[:, 1]
        return np.clip(self.calibrator.predict(raw), PROB_FLOOR, PROB_CEIL)

    def _interval(self, X: pd.DataFrame, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Union of two honest, different uncertainties:
        estimation variance (bootstrap over training resamples, which widens for
        unusual inputs) and calibration sampling error (Wilson on the calibration
        fold's realised frequency, which widens where evidence is thin)."""
        lo, hi = p.copy(), p.copy()
        boots = self._attr("bootstrap_models", [])
        if boots:
            samples = np.vstack([
                np.clip(cal_i.predict(bst_i.predict_proba(X[self.feature_columns])[:, 1]), 0.0, 1.0)
                for bst_i, cal_i in boots
            ])
            lo = np.minimum(lo, np.percentile(samples, CI_LOW_PCT, axis=0))
            hi = np.maximum(hi, np.percentile(samples, CI_HIGH_PCT, axis=0))
        edges = self._attr("calib_edges", None)
        if edges is not None and len(edges) >= 2:
            b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
            lo = np.minimum(lo, np.asarray(self.calib_low)[b])
            hi = np.maximum(hi, np.asarray(self.calib_high)[b])
        return np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)

    def _comparables(self, df: pd.DataFrame, n: int) -> tuple[np.ndarray, list, np.ndarray]:
        """BRD guardrail input, with the cold-start hierarchy productID ->
        product_type -> none. ``n_comparable`` stays the product's own count so
        the guardrail keeps its literal BRD meaning; the fallback only decides
        which Confidence bucket a brand-new product lands in."""
        gc = self.group_col
        keys = df[gc].astype(str) if (gc and gc in df.columns) else pd.Series([None] * n)
        own = np.array([self.product_support.get(k, 0) for k in keys])

        tsup = self._attr("product_type_support", {})
        if tsup and "product_type" in df.columns:
            tkeys = df["product_type"].astype(str)
            fallback = np.array([tsup.get(k, 0) for k in tkeys])
        else:
            fallback = np.zeros(n, dtype=int)

        eff = np.where(own > 0, own, fallback)
        level = ["product" if o > 0 else ("product_type" if f > 0 else "none")
                 for o, f in zip(own, fallback)]
        return own, level, eff

    def _sweep_bounds(self) -> tuple | None:
        """The range the band may be swept over, and the range a caller's price is
        judged against: the INFORMATIVE range where the response moves, not merely
        the observed one. Falls back to price_bounds for champions pickled before
        this existed, so an old model keeps predicting."""
        b = self._attr("price_informative_bounds", None)
        if b is not None and np.isfinite([b[0], b[1]]).all() and b[1] > b[0]:
            return float(b[0]), float(b[1])
        return self._attr("price_bounds", None)

    def _price_out_of_range(self, df: pd.DataFrame, n: int) -> np.ndarray:
        """Is the incoming price_ratio outside the range this model can SCORE?

        Outside it the monotone constraint returns a constant — the ceiling below
        the bottom, the floor above the top — so the number stops carrying
        information while still looking like a normal answer. This used to test
        against observed pricing, which was too generous: there was a stretch
        inside the observed range where the response was already flat and the flag
        stayed false. It now tests the informative range, so flat means flagged."""
        bounds = self._sweep_bounds()
        if bounds is None or PRICE_FEATURE not in df.columns:
            return np.zeros(n, dtype=bool)
        r = pd.to_numeric(df[PRICE_FEATURE], errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(r), (r < bounds[0]) | (r > bounds[1]), False)

    def _score_at(self, X: pd.DataFrame, ratios: np.ndarray) -> np.ndarray:
        """Served probability for every row at its own price ratio. One batched pass."""
        probe = X.copy()
        probe[PRICE_FEATURE] = ratios
        if BELOW_COST in probe.columns:
            probe[BELOW_COST] = (ratios < 1.0).astype(float)
        return self._calibrated(probe)

    def _refine_edges(self, X: pd.DataFrame, r_in: np.ndarray, r_out: np.ndarray,
                      thresh: np.ndarray, iters: int = 14) -> np.ndarray:
        """Per row: the largest ratio in [r_in, r_out] whose probability is still
        >= thresh. Bisection, all rows in lockstep — `iters` batched scoring passes
        for the whole request, not per line.

        This exists because the sweep grid is coarse relative to the calibrator's
        step function. Isotonic regression emits a step response, and a plateau can
        be a few cents wide while one grid step is a quarter of a dollar. Reporting
        the last GRID point inside the band truncates it, and when the step falls
        between the first two eligible points the band degenerates to
        low == mid == high — "$22.14 – $22.14", which reads as a broken answer
        rather than as the true edge of a narrow plateau.
        """
        lo = np.array(r_in, dtype=float).copy()
        hi = np.maximum(np.array(r_out, dtype=float), lo)
        live = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
        if not live.any():
            return lo
        for _ in range(iters):
            mid = np.where(live, (lo + hi) / 2.0, lo)
            p = self._score_at(X, np.nan_to_num(mid, nan=1.0))
            ok = live & (p >= thresh)
            lo = np.where(ok, mid, lo)
            hi = np.where(live & ~ok, mid, hi)
        return lo

    def _price_band(self, df: pd.DataFrame, X: pd.DataFrame, n: int) -> pd.DataFrame:
        """Recommended Price Band = the prices that MAXIMISE WIN PROBABILITY, at or
        above cost, swept only where the model has evidence.

        Because P(win) is monotone non-increasing in price, the best odds always sit
        at the cheapest price we are willing to quote — which is cost. So:

            floor  = max(cost / basis, bottom of the informative range)
            p_max  = P(win) at that floor        <- the best odds available
            band   = prices running UP from the floor while P(win) >= p_max - PROB_BAND_TOL
            mid    = the highest price still within PROB_BAND_TOL / 2 of p_max

        Every price in the band wins within five points as often as the best price
        does. `mid` takes the most margin available for effectively unchanged odds,
        which is not a trade-off — it is the same win rate for more money.

        The expected-margin optimum is computed too and returned as `ev_price_*`.
        It is NOT the recommendation: EV pushes the price up until the odds start
        paying for the margin, which answers a different question than "what price
        is most likely to be accepted".
        """
        # Typed up front (nan / "" rather than None) so the logged MLflow signature
        # infers a stable schema even when no band can be computed.
        low_a, mid_a, high_a, cur_a = (np.full(n, np.nan) for _ in range(4))
        # the recommended mid expressed back as a ratio, so predict() can score the
        # model AT its own recommendation without recomputing the basis
        ratio_mid_a = np.full(n, np.nan)
        # the odds AT each edge of the band, so the panel can say what the rep gets
        p_low_a, p_mid_a, p_high_a = (np.full(n, np.nan) for _ in range(3))
        # the expected-margin optimum, for comparison only — never the recommendation
        ev_low_a, ev_mid_a, ev_high_a = (np.full(n, np.nan) for _ in range(3))
        status = np.array(["unavailable"] * n, dtype=object)
        message = np.array([""] * n, dtype=object)
        clamped_a, ood_a, edge_a = (np.zeros(n, dtype=bool) for _ in range(3))

        def assemble():
            return pd.DataFrame({
                "recommended_price_low": low_a, "recommended_price_mid": mid_a,
                "recommended_price_high": high_a, "current_price": cur_a,
                "price_band_status": status, "price_band_message": message,
                "price_clamped": clamped_a, "price_band_out_of_distribution": ood_a,
                "boundary_hit": edge_a, "recommended_ratio_mid": ratio_mid_a,
                # what the band is optimised FOR, stated in the payload so nobody
                # has to infer it from the numbers
                "band_objective": "max_win_probability_above_cost",
                "win_probability_at_low_pct": np.round(p_low_a * 100, 1),
                "win_probability_at_mid_pct": np.round(p_mid_a * 100, 1),
                "win_probability_at_high_pct": np.round(p_high_a * 100, 1),
                "ev_price_low": ev_low_a, "ev_price_mid": ev_mid_a,
                "ev_price_high": ev_high_a,
            }, index=df.index)

        # Swept over the INFORMATIVE range only. This is the fix for the panel
        # recommending a price inside the flat region: the grid cannot reach there,
        # so neither can the recommendation.
        bounds = self._sweep_bounds()
        if bounds is None or PRICE_FEATURE not in self.feature_columns or COST_FEATURE not in df.columns:
            return assemble()

        r_lo, r_hi = float(bounds[0]), float(bounds[1])
        if not np.isfinite([r_lo, r_hi]).all() or r_hi <= r_lo:
            return assemble()

        cost = pd.to_numeric(df[COST_FEATURE], errors="coerce").to_numpy(dtype=float)
        basis = (pd.to_numeric(df[BASIS_FEATURE], errors="coerce").to_numpy(dtype=float)
                 if BASIS_FEATURE in df.columns else cost)
        basis = np.where(np.isfinite(basis) & (basis > 0), basis, cost)
        qty = (pd.to_numeric(df["quantity"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
               if "quantity" in df.columns else np.ones(n))
        cur_r = pd.to_numeric(df.get(PRICE_FEATURE), errors="coerce").to_numpy(dtype=float) \
            if PRICE_FEATURE in df.columns else np.full(n, np.nan)
        base = (pd.to_numeric(df["base_price"], errors="coerce").to_numpy(dtype=float)
                if "base_price" in df.columns else np.full(n, np.nan))

        grid = np.linspace(r_lo, r_hi, SWEEP_POINTS)
        # One batched scoring pass: n x SWEEP_POINTS rows. ONLY price_ratio moves.
        #
        # `quote_total` in particular is held at the value that came in, and must
        # stay that way. It is a genuine feature and it is derived from the prices
        # the caller typed, so it is tempting to move it with the swept price —
        # "if they quoted 1200 instead of 1000, the quote total would be higher too".
        # Do not. The monotone constraint covers price_ratio and NOTHING ELSE, so a
        # second price-driven feature moving alongside it can push the net response
        # back UP, and the panel would be telling a rep that raising the price
        # improves their odds. Holding every other feature fixed is what makes the
        # sweep a clean partial derivative in price, which is the only thing the
        # constraint can guarantee.
        #
        # `below_cost` is the one exception: it is a deterministic function of the
        # swept price, not an independent feature, so leaving it stale would score
        # a below-cost price as though it were above cost.
        rep = X.loc[X.index.repeat(SWEEP_POINTS)].reset_index(drop=True)
        tiled = np.tile(grid, n)
        rep[PRICE_FEATURE] = tiled
        if BELOW_COST in rep.columns:
            rep[BELOW_COST] = (tiled < 1.0).astype(float)
        p_grid = self._calibrated(rep).reshape(n, SWEEP_POINTS)

        # collected per row, refined in one batched pass after the loop
        r_low_g = np.full(n, np.nan)
        hi_in, hi_out = np.full(n, np.nan), np.full(n, np.nan)
        mid_in, mid_out = np.full(n, np.nan), np.full(n, np.nan)
        thr_hi, thr_mid = np.full(n, -np.inf), np.full(n, -np.inf)
        b_arr = np.full(n, np.nan)
        band_ok = np.zeros(n, dtype=bool)

        for i in range(n):
            c = cost[i]
            if not np.isfinite(c) or c <= 0:
                continue
            b_i = basis[i] if np.isfinite(basis[i]) and basis[i] > 0 else c
            p_i = p_grid[i]

            # --- the floor: never below cost ---------------------------------
            # The best odds sit at the cheapest acceptable price, so the floor IS
            # the probability optimum. Below cost the company loses money on a win,
            # which is not a recommendation whatever the odds say.
            r_floor = c / b_i
            elig = grid >= r_floor - 1e-12
            if not elig.any():
                # every scoreable price is below cost — say so rather than
                # recommending a loss or silently returning nothing
                status[i] = "unavailable"
                message[i] = ("No price at or above cost is inside the range this model "
                              "can score — this line cannot be priced profitably on the "
                              "evidence available")
                continue
            first = int(np.argmax(elig))          # lowest eligible grid index

            # --- the band: prices whose odds are within tolerance of the best --
            p_max = float(p_i[first])
            keep = p_i >= p_max - PROB_BAND_TOL
            hi_i = first
            while hi_i < SWEEP_POINTS - 1 and elig[hi_i + 1] and keep[hi_i + 1]:
                hi_i += 1
            # mid: the dearest price still within HALF the tolerance. Same odds to
            # within ~2.5 points, more margin — free, not a trade-off.
            mid_i = first
            while (mid_i < hi_i and p_i[mid_i + 1] >= p_max - PROB_BAND_TOL / 2.0):
                mid_i += 1

            # Edges are refined off the grid AFTER this loop, in one batched pass —
            # record where each one sits between grid points.
            r_low_g[i] = grid[first]
            hi_in[i], hi_out[i] = grid[hi_i], (grid[hi_i + 1] if hi_i < SWEEP_POINTS - 1
                                               else grid[hi_i])
            mid_in[i], mid_out[i] = grid[mid_i], (grid[mid_i + 1] if mid_i < SWEEP_POINTS - 1
                                                  else grid[mid_i])
            thr_hi[i] = p_max - PROB_BAND_TOL
            thr_mid[i] = p_max - PROB_BAND_TOL / 2.0
            b_arr[i] = b_i
            band_ok[i] = True
            # The band running to the top of the scoreable range is meaningful, not a
            # defect: it says the odds never fall away inside the evidence we have.
            edge_a[i] = bool(hi_i == SWEEP_POINTS - 1)

            # --- expected margin, for comparison only ------------------------
            margin = (grid * b_i - c) * (qty[i] if np.isfinite(qty[i]) else 1.0)
            ev = np.where(elig, p_i * margin, np.nan)
            if np.isfinite(ev).any() and np.nanmax(ev) > 0:
                peak = int(np.nanargmax(ev))
                ekeep = np.nan_to_num(ev, nan=-np.inf) >= (1.0 - EV_BAND_TOL) * ev[peak]
                e_lo, e_hi = peak, peak
                while e_lo > 0 and ekeep[e_lo - 1]:
                    e_lo -= 1
                while e_hi < SWEEP_POINTS - 1 and ekeep[e_hi + 1]:
                    e_hi += 1
                ev_low_a[i] = round(grid[e_lo] * b_i, 2)
                ev_mid_a[i] = round(grid[peak] * b_i, 2)
                ev_high_a[i] = round(grid[e_hi] * b_i, 2)

        # --- refine both edges off the grid, all rows at once -----------------
        r_high = self._refine_edges(X, hi_in, np.minimum(hi_out, r_hi), thr_hi)
        r_mid = self._refine_edges(X, mid_in, np.minimum(mid_out, r_high), thr_mid)
        # odds AT the refined edges, so the reported percentages match the prices
        # shown rather than the grid points they were found from
        safe = np.where(np.isfinite(r_low_g), r_low_g, 1.0)
        p_at_low = self._score_at(X, safe)
        p_at_mid = self._score_at(X, np.where(np.isfinite(r_mid), r_mid, 1.0))
        p_at_high = self._score_at(X, np.where(np.isfinite(r_high), r_high, 1.0))

        for i in range(n):
            if not band_ok[i]:
                continue
            b_i, c = b_arr[i], cost[i]
            low, mid, high = r_low_g[i] * b_i, r_mid[i] * b_i, r_high[i] * b_i
            p_low_a[i], p_mid_a[i], p_high_a[i] = p_at_low[i], p_at_mid[i], p_at_high[i]

            if np.isfinite(base[i]) and base[i] > 0:      # BRD +/-30% of base Sales Price
                clamped_a[i] = bool(is_price_clamped(low, base[i])
                                    or is_price_clamped(high, base[i]))
                low = float(clamp_price(low, base[i]))
                high = float(clamp_price(high, base[i]))
                mid = float(clamp_price(mid, base[i]))
                # the clamp may push the floor below cost; the cost rule wins
                low = max(low, round(c, 2))
                mid, high = max(mid, low), max(high, low)

            low_a[i], mid_a[i], high_a[i] = round(low, 2), round(mid, 2), round(high, 2)
            if b_i > 0:
                ratio_mid_a[i] = mid / b_i

            if np.isfinite(cur_r[i]):
                cur = float(cur_r[i]) * b_i
                cur_a[i] = round(cur, 2)
                within = low_a[i] <= cur <= high_a[i]
                status[i] = "within_range" if within else "out_of_range"
                message[i] = f"Current ${cur:,.2f} {'within' if within else 'out of'} range"
                ood_a[i] = bool(cur_r[i] < r_lo or cur_r[i] > r_hi)
        return assemble()

    # -- pyfunc contract ----------------------------------------------------
    def predict(self, context, model_input, params=None):
        df = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        df = df.reset_index(drop=True)
        n = len(df)
        X = self._coerce(df)
        p = self._calibrated(X)
        ci_lo, ci_hi = self._interval(X, p)
        n_comp, level, n_eff = self._comparables(df, n)

        conf = [confidence_level(int(c)) for c in n_eff]
        hidden = n_comp < MIN_COMPARABLE                       # BRD: hide the score
        score_edge = (p < LOW_CONF_LOW) | (p > LOW_CONF_HIGH)  # BRD: extreme score
        price_ood = self._price_out_of_range(df, n)
        low_conf = hidden | price_ood | score_edge
        # ordered by how much each invalidates the number: no evidence, then a
        # price we cannot score, then a merely extreme score
        reason = np.where(hidden, "few_comparable",
                          np.where(price_ood, "price_out_of_range",
                                   np.where(score_edge, "score_out_of_range", "")))
        message = np.where(hidden, MSG_FEW_COMPARABLE,
                           np.where(price_ood, MSG_PRICE_RANGE,
                                    np.where(score_edge, MSG_SCORE_RANGE, MSG_SUFFICIENT)))

        band = np.where(p >= PROB_BAND_HIGH, "High", np.where(p >= PROB_BAND_MED, "Medium", "Low"))
        label = np.array([f"{b} likelihood of conversion" for b in band])

        # The price band, and the model's own probability AT the price it
        # recommends. The recommendation is now always inside the informative
        # range, so this number is always a real score even when the caller's own
        # price is not — which is what the panel should be showing.
        pb = self._price_band(df, X, n)
        r_mid = pd.to_numeric(pb["recommended_ratio_mid"], errors="coerce").to_numpy(float)
        p_rec = np.full(n, np.nan)
        ok = np.isfinite(r_mid)
        if ok.any() and PRICE_FEATURE in self.feature_columns:
            Xr = X.copy()
            Xr.loc[ok, PRICE_FEATURE] = r_mid[ok]
            if BELOW_COST in Xr.columns:      # keep the derived flag consistent
                Xr.loc[ok, BELOW_COST] = (r_mid[ok] < 1.0).astype(float)
            p_rec[ok] = self._calibrated(Xr)[ok]

        out = pd.DataFrame({
            # --- unchanged serving contract ---
            "win_probability": np.round(p, 4),
            "win_probability_pct": np.round(p * 100, 1),
            "win_probability_ci_low": np.round(ci_lo, 4),
            "win_probability_ci_high": np.round(ci_hi, 4),
            "win_probability_ci_low_pct": np.round(ci_lo * 100, 1),
            "win_probability_ci_high_pct": np.round(ci_hi * 100, 1),
            "n_comparable": n_comp,
            "confidence": conf,
            # --- Win Probability card ---
            # `win_probability` always carries the number (the serving contract
            # never goes null); `win_probability_display` is what the panel binds
            # to, and is NaN when the BRD says show the guardrail card INSTEAD of
            # a score. Showing a confident 85% next to "few comparable quotes"
            # means the 85% wins and the warning is decoration.
            # ALSO blanked when the caller's price sits in the dead zone. A flat
            # 89.9% shown beside "price outside the range this model has seen" is
            # the warning losing to the number every time.
            "win_probability_display": np.where(hidden | price_ood, np.nan,
                                                np.round(p * 100, 1)),
            # Is the number above a score, or the constant the booster returns
            # where it has no splits? Bind any "trust this" UI to THIS field.
            "win_probability_reliable": ~(hidden | price_ood),
            # The odds at the recommended mid — always a real score, because the
            # recommendation can no longer land outside the informative range.
            # This is what the panel should lead with when the rep's own price
            # cannot be scored.
            "win_probability_at_recommended": np.round(p_rec, 4),
            "win_probability_at_recommended_pct": np.round(p_rec * 100, 1),
            "probability_band": band,
            "probability_label": label,
            # --- Confidence Level card ---
            "hidden": hidden,
            "low_confidence": low_conf,
            "confidence_label": np.where(low_conf, "Low Confidence", "High Confidence"),
            "confidence_message": message,
            "guardrail_reason": reason,
            "n_comparable_effective": n_eff,
            "support_level": level,
            "price_input_out_of_distribution": price_ood,
            # "unknown", not a plausible-looking default. A champion pickled before
            # this field existed genuinely does not know its basis, and reporting
            # "unitPrice" sent the last debugging session the wrong way.
            "price_basis": self._attr("price_basis", "unknown"),
            "model_provenance": self._attr("provenance", "csv"),
        })
        return pd.concat([out, pb], axis=1)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def train(df_or_path, tenant: str, *, auto_hpo: bool = True, register: bool = True,
          source: str = "csv", logger: RunLogger | None = None) -> dict:
    logger = logger or RunLogger()
    logger.set_progress(10, "loading_data", "Loading training data")
    df = pd.read_csv(df_or_path) if isinstance(df_or_path, str) else df_or_path.copy()
    df = df.reset_index(drop=True)

    # `tenant` names the registered model only — never a feature or a row filter
    # (see csv_win.train). Train on the whole frame (pooled = GLOBAL base model).
    # NOTE: the BRD NFR reads "model is trained and scored per tenant only". This
    # pooled base model contradicts that as written and needs the NFR amended to
    # "global base, tenant-scoped calibration and serving" before sign-off.
    if "tenant" in df.columns:
        present = sorted(map(str, df["tenant"].dropna().unique()))
        logger.log(f"'tenant' column present {present} — ignored for modelling; training on "
                   f"all {len(df)} lines (tenant only names the model '{tenant}')")
    else:
        logger.log(f"Loaded {len(df)} lines (source={source}, model tenant={tenant})")

    if LABEL not in df.columns:
        raise ValueError(f"dataset missing required columns: ['{LABEL}']")

    df = _with_derived(df)
    if BELOW_COST in df.columns:
        probe = pd.to_numeric(df[LABEL], errors="coerce").fillna(0).astype(int)
        if not _below_cost_helps(df, probe, logger):
            df = df.drop(columns=[BELOW_COST])
    num, cat = _select_features(df)
    features = num + cat
    dropped = [c for c in LINE_NUMERIC + BRD_NUMERIC + HISTORY_NUMERIC + DERIVED_NUMERIC
               + LINE_CATEG + BRD_CATEG if c not in features]
    if len(features) < 2:
        raise ValueError(f"only {len(features)} usable feature(s) present — cannot train")
    have, missing = _brd_coverage(df)
    logger.log(f"BRD training data elements: {len(have)}/{len(BRD_ELEMENTS)} available"
               + (f"  |  MISSING (skipped, not synthesized): {missing}" if missing else ""))
    logger.log(f"Using {len(features)} features: {features}"
               + (f"  |  not present: {dropped}" if dropped else ""))

    # Generic pass shared by every module (maxxflow_features.cleaning):
    # DEDUPE -> WINSORISE -> IMPUTE.
    #
    # The dedupe key includes the QUOTATION, so an exact duplicate is only ever
    # collapsed inside one quote — that is an export artefact and double-weights the
    # pattern. Two DIFFERENT quotes that happen to price the same product identically
    # are two real observations of the market and both are kept; deduping across
    # quotes would quietly delete evidence and change the grouped split underneath us.
    # The label is in the key too: identical features with opposite outcomes are real
    # label noise the model must see.
    #
    # `won` is not in `num`, so the label is never winsorised. PRICE_FEATURE is held
    # out of the clip deliberately: its tails are the decision. Deep discounts and
    # premium pricing are precisely what the panel is asked about, and clipping them
    # to the 0.5-99.5% band measurably costs AUC (0.6212 -> 0.6186 on the all-verticals
    # export, 0.7809 -> 0.7767 on the price-driven fixture). It also does not need the
    # protection: price_bounds below already tells a caller when an incoming ratio is
    # off the trained scale, which is the honest response to an extreme price.
    dedupe_key = ([QUOTE_COL] if QUOTE_COL in df.columns else []) + num + cat + [LABEL]
    df, dq = clean_frame(df, num, cat, logger, dedupe_subset=dedupe_key,
                         winsorize=[c for c in num if c != PRICE_FEATURE])
    logger.log(f"Review & Clean: {dq.summary()}")
    logger.set_progress(20, "cleaning_data", "Review and clean complete")

    y = pd.to_numeric(df[LABEL], errors="coerce").fillna(0).astype(int)
    if y.nunique() < 2:
        raise ValueError(f"tenant {tenant}: only one class present — cannot train a classifier")
    X = df[features].copy()
    for c in cat:
        X[c] = X[c].astype("category")
    # Take the fill values from the FITTED stats rather than recomputing them off the
    # already-filled frame. Same numbers, but sourced from the one place that applied
    # them, so training and serving cannot drift apart. A column clean_frame could not
    # fit (all-missing) falls back to 0.0 instead of NaN, which would poison _coerce.
    _medians = (dq.stats.get("impute") or {}).get("medians") or {}
    numeric_fill = {}
    for c in num:
        v = _medians.get(c, pd.to_numeric(df[c], errors="coerce").median())
        numeric_fill[c] = float(v) if pd.notna(v) else 0.0

    group_col = GROUP if GROUP in df.columns else None
    if group_col and QUOTE_COL in df.columns:
        # BRD counts comparable QUOTATIONS, not lines: one quote carrying the same
        # product on five lines is one comparable.
        product_support = df.groupby(df[group_col].astype(str))[QUOTE_COL].nunique().to_dict()
        support_basis = "quotations"
    elif group_col:
        product_support = df[group_col].astype(str).value_counts().to_dict()
        support_basis = "lines"
    else:
        product_support, support_basis = {}, "lines"
    if "product_type" in df.columns and QUOTE_COL in df.columns:
        type_support = df.groupby(df["product_type"].astype(str))[QUOTE_COL].nunique().to_dict()
    elif "product_type" in df.columns:
        type_support = df["product_type"].astype(str).value_counts().to_dict()
    else:
        type_support = {}
    logger.log(f"Comparables counted in {support_basis}: {len(product_support)} products, "
               f"median {int(np.median(list(product_support.values()))) if product_support else 0} "
               f"per product (BRD guardrail fires below {MIN_COMPARABLE})")

    i_tr, i_ca, i_te, strategy = _three_way_split(df, y, logger)
    Xtr, ytr = X.iloc[i_tr], y.iloc[i_tr]
    Xca, yca = X.iloc[i_ca], y.iloc[i_ca]
    Xte, yte = X.iloc[i_te], y.iloc[i_te]
    logger.log(f"Folds: {len(Xtr)} fit / {len(Xca)} calibrate / {len(Xte)} score. Calibration is "
               f"fitted on its OWN fold so Brier/ECE — the metrics the promotion gate reads — "
               f"are measured on data the calibrator never saw.")
    logger.set_progress(25, "splitting_data", "Training folds prepared")

    params = auto_tune_classifier(Xtr, ytr, FINALIZED, logger) if auto_hpo else dict(FINALIZED)
    if not auto_hpo:
        logger.set_progress(55, "hyperparameter_search", "Using configured parameters")
    mono = _monotone_constraints(features)
    fit_params = {**params, "monotone_constraints": mono}
    if PRICE_FEATURE in features:
        logger.log(f"Monotone constraint -1 on '{PRICE_FEATURE}': raising price can never raise "
                   f"the displayed win probability, whatever the endogeneity in the data says.")

    # Early stopping needs its own validation signal — reuse the calibration fold
    # (already held out from Xte) rather than manufacturing a fourth split. Below
    # EARLY_STOP_MIN_ROWS the AUC eval metric is too noisy to trust (see module
    # constants), so fall back to the fixed, tuned n_estimators exactly as before.
    use_early_stop = len(Xca) >= EARLY_STOP_MIN_ROWS and yca.nunique() >= 2
    logger.set_progress(60, "fitting_model", "Fitting the primary model")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if use_early_stop:
            tuned_n = fit_params.pop("n_estimators")
            booster = LGBMClassifier(verbose=-1, random_state=7,
                                     n_estimators=max(tuned_n, EARLY_STOP_MAX_ROUNDS),
                                     **fit_params).fit(
                Xtr, ytr, eval_set=[(Xca, yca)], eval_metric="auc",
                callbacks=[early_stopping(EARLY_STOP_ROUNDS, verbose=False)])
            fit_params["n_estimators"] = booster.best_iteration_ or tuned_n
            logger.log(f"Early stopping: best iteration {fit_params['n_estimators']} on the "
                       f"calibration fold (tuned n_estimators was {tuned_n}, cap "
                       f"{max(tuned_n, EARLY_STOP_MAX_ROUNDS)}) — stopped after "
                       f"{EARLY_STOP_ROUNDS} rounds without AUC improvement. The bootstrap "
                       f"ensemble below reuses this capacity instead of early-stopping per model.")
        else:
            booster = LGBMClassifier(verbose=-1, random_state=7, **fit_params).fit(Xtr, ytr)
            logger.log("Early stopping skipped — calibration fold too small or single-class for "
                       "a validation signal; using the tuned n_estimators fixed.")
    logger.set_progress(65, "calibrating_model", "Calibrating model probabilities")

    raw_ca = booster.predict_proba(Xca)[:, 1]
    if len(Xca) >= MIN_FOLD_ROWS and yca.nunique() >= 2:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=PROB_FLOOR,
                                 y_max=PROB_CEIL).fit(raw_ca, yca)
    else:
        iso = _IdentityCalibrator()
        logger.log("Calibration fold too small or single-class — serving UNCALIBRATED scores; "
                   "the displayed percentage is a ranking, not a frequency.")
    p_ca = np.clip(iso.predict(raw_ca), PROB_FLOOR, PROB_CEIL)
    edges, c_lo, c_hi = _calibration_bands(p_ca, yca.to_numpy())

    # Score the holdout exactly as the model will serve it, floor/ceiling included.
    p = np.clip(iso.predict(booster.predict_proba(Xte)[:, 1]), PROB_FLOOR, PROB_CEIL)
    pred = (p >= 0.5).astype(int)

    # Bootstrap ensemble: resample the FIT fold only, calibrate each against the
    # calibration fold (never the scoring holdout), reusing the tuned params.
    rng = np.random.RandomState(7)
    bootstrap_models = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(N_BOOTSTRAP):
            idx = rng.randint(0, len(Xtr), len(Xtr))
            Xb, yb = Xtr.iloc[idx], ytr.iloc[idx]
            if yb.nunique() < 2:
                logger.set_progress(
                    65 + (i + 1) * 20 / N_BOOTSTRAP,
                    "bootstrap_ensemble",
                    f"Bootstrap model {i + 1} of {N_BOOTSTRAP}",
                    current=i + 1,
                    total=N_BOOTSTRAP,
                    message=f"Bootstrap model {i + 1}/{N_BOOTSTRAP} skipped (single class)",
                )
                continue
            bst_i = LGBMClassifier(verbose=-1, random_state=7 + i + 1, **fit_params).fit(Xb, yb)
            r_i = bst_i.predict_proba(Xca)[:, 1]
            cal_i = (IsotonicRegression(out_of_bounds="clip", y_min=PROB_FLOOR,
                                        y_max=PROB_CEIL).fit(r_i, yca)
                     if yca.nunique() >= 2 else _IdentityCalibrator())
            bootstrap_models.append((bst_i, cal_i))
            logger.set_progress(
                65 + (i + 1) * 20 / N_BOOTSTRAP,
                "bootstrap_ensemble",
                f"Bootstrap model {i + 1} of {N_BOOTSTRAP}",
                current=i + 1,
                total=N_BOOTSTRAP,
                message=f"Bootstrap model {i + 1}/{N_BOOTSTRAP} fitted",
            )
    logger.log(f"Fit {len(bootstrap_models)}/{N_BOOTSTRAP} bootstrap models; displayed interval is "
               f"the union of that spread and the Wilson interval on the calibration fold's "
               f"realised win rate.")

    price_bounds, price_basis = None, "unitPrice"
    if BASIS_FEATURE in df.columns and COST_FEATURE in df.columns:
        b = pd.to_numeric(df[BASIS_FEATURE], errors="coerce")
        c = pd.to_numeric(df[COST_FEATURE], errors="coerce")
        price_basis = "listPrice" if not np.allclose(b.fillna(0), c.fillna(0)) else "unitPrice"
    if PRICE_FEATURE in features:
        s = pd.to_numeric(df[PRICE_FEATURE], errors="coerce").dropna()
        if len(s):
            price_bounds = (float(np.percentile(s, OBS_LO_PCT)), float(np.percentile(s, OBS_HI_PCT)))
            logger.log(f"price_ratio is measured against {price_basis} — a caller that sends "
                       f"a ratio on a different basis will be scored on the wrong scale")
            logger.log(f"Price sweep clamped to observed pricing: {PRICE_FEATURE} in "
                       f"[{price_bounds[0]:.4f}, {price_bounds[1]:.4f}] — the band is never "
                       f"recommended where the model has no evidence.")

    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    base_rate = float(max(yte.mean(), 1 - yte.mean()))   # accuracy of always guessing the majority
    metrics = {
        "accuracy": float((pred == yte.to_numpy()).mean()),
        "auc": float(roc_auc_score(yte, p)), "pr_auc": float(average_precision_score(yte, p)),
        "precision": float(precision_score(yte, pred, zero_division=0)),
        "recall": float(recall_score(yte, pred, zero_division=0)),
        "f1": float(f1_score(yte, pred, zero_division=0)),
        "brier": float(brier_score_loss(yte, p)), "ece": float(expected_calibration_error(yte, p)),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)), "n_calibration": int(len(Xca)),
        "positive_rate": float(y.mean()),
        # A 72%-base-rate problem makes raw accuracy look good for free. Report the
        # majority-class baseline next to it so nobody reads 0.72 as skill.
        "base_rate_accuracy": base_rate,
        "accuracy_over_base_rate": float((pred == yte.to_numpy()).mean() - base_rate),
    }
    logger.log(f"Metrics ({strategy} holdout): accuracy={metrics['accuracy']:.3f} "
               f"(majority-class baseline {base_rate:.3f}) AUC={metrics['auc']:.3f} "
               f"F1={metrics['f1']:.3f} Brier={metrics['brier']:.3f} ECE={metrics['ece']:.3f}")
    logger.set_progress(90, "evaluating_model", "Evaluation complete")

    categories = {c: list(X[c].cat.categories) for c in cat}
    model = LineWinModel(booster, iso, categories, features, product_support, group_col,
                         provenance=source, bootstrap_models=bootstrap_models,
                         numeric_fill=numeric_fill, calib_edges=edges, calib_low=c_lo,
                         calib_high=c_hi, price_bounds=price_bounds,
                         product_type_support=type_support, support_basis=support_basis,
                         price_basis=price_basis)

    # Ask the fitted model where its own answer actually moves, and keep the band
    # inside that. Measured AFTER the model is assembled because it is a property
    # of the served response — calibration included — not of the booster alone.
    model.price_informative_bounds = _informative_price_range(model, df.iloc[i_te], logger)

    # Behavioural check, not a unit test: sweep real holdout rows and assert the
    # served price response never rises. If this ever trips, the panel is telling
    # reps that putting the price up improves their odds.
    metrics["price_monotonicity_violated"] = float(_monotonicity_violation(model, df.iloc[i_te]))
    metrics["price_monotonicity_violated_full_range"] = float(
        _monotonicity_violation(model, df.iloc[i_te], above_cost_only=False))
    logger.log(f"Served price response rises on {metrics['price_monotonicity_violated']:.1%} of "
               f"sampled holdout lines at or above cost "
               f"({metrics['price_monotonicity_violated_full_range']:.1%} across the full sweep, "
               f"where crossing out of below-cost pricing may legitimately help)")

    result = {"model_type": "classification", "metrics": metrics, "features": features,
              "dropped_features": dropped, "source": source,
              "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
              "params": params, "logs": logger.lines,
              "split_strategy": strategy, "monotone_constraints": dict(zip(features, mono)),
              "price_ratio_bounds": price_bounds, "price_basis": price_basis,
              "price_informative_bounds": model.price_informative_bounds,
              "support_basis": support_basis,
              "data_quality": {"rows_in": dq.rows_in, "rows_out": dq.rows_out,
                               "duplicates_removed": dq.duplicates_removed,
                               "missing_filled": dq.missing_filled,
                               "outliers_clipped": dq.outliers_clipped},
              "brd_elements_present": have, "brd_elements_missing": missing}
    if not register:
        logger.set_progress(100, "complete", "Training complete")
        result["_model"] = model
        return result

    logger.set_progress(95, "registering_candidate", "Registering candidate model")
    reg = MLflowRegistry()
    name = registered_model_name(tenant, MODULE)
    tags = {"tenant": tenant, "module": MODULE, "data_provenance": source,
            "accuracy": f"{metrics['accuracy']:.5f}", "auc": f"{metrics['auc']:.5f}",
            "brier": f"{metrics['brier']:.5f}", "ece": f"{metrics['ece']:.5f}",
            "split_strategy": strategy,
            # data-quality lineage: each registered version records what the
            # cleaning pass actually did to the data it was fitted on.
            **dq.as_tags()}
    # below_cost is derived at serve time, never an input.
    # BASIS_FEATURE is not a MODEL feature — the booster never sees it — but the
    # price band needs it to turn a ratio back into money. It has to be in the
    # SIGNATURE regardless, because MLflow enforces the schema before predict()
    # runs and silently DROPS any column the schema does not name. Leaving it out
    # is why the served band came back on a cost scale while the same model,
    # called directly, returned a list-price band: through the API the column
    # never arrived, and _price_band fell back to unit cost.
    present_raw = [c for c in (LINE_NUMERIC + BRD_NUMERIC + HISTORY_NUMERIC
                              + LINE_CATEG + BRD_CATEG + [BASIS_FEATURE])
                   if c in df.columns]
    ex = df[present_raw].head(3).copy()
    for c in present_raw:
        ex[c] = (ex[c].astype(str) if c in (LINE_CATEG + BRD_CATEG)
                 else pd.to_numeric(ex[c], errors="coerce").astype(float))
    sig = _build_signature(ex, model.predict(None, ex))
    version = reg.log_and_register(model, name=name, params={"algo": "lightgbm+isotonic", **params},
                                   metrics=metrics, tags=tags, signature=sig, input_example=ex)
    logger.log(f"Registered {name} v{version} (candidate — not yet champion)")
    logger.set_progress(98, "candidate_registered", f"Candidate v{version} registered")
    result.update({"registered_name": name, "version": version, "champion": _champion_metrics(reg, name)})
    return result


def _informative_price_range(model: LineWinModel, sample: pd.DataFrame,
                             logger: RunLogger | None = None) -> tuple | None:
    """Where does the served price response actually MOVE?

    A booster has no split below its lowest split point on price_ratio, so every
    price under it lands in the same leaf and returns the same probability. That
    region looks like a confident answer and contains no information — and the
    expected-margin sweep, left to itself, will happily put the recommended band
    there, because a flat win term makes the cheapest price look free of risk.

    So probe it: hold every other feature fixed, walk price_ratio across observed
    pricing, and take the median served probability at each step. The informative
    range is the stretch where that median differs from its own plateaus by more
    than FLAT_TOL. Outside it the model is returning a constant and should say so.

    Returns None when it cannot be measured, in which case the caller keeps using
    the observed bounds — the previous behaviour, no worse than before.
    """
    bounds = getattr(model, "price_bounds", None)
    if bounds is None or PRICE_FEATURE not in model.feature_columns or sample.empty:
        return None
    lo, hi = float(bounds[0]), float(bounds[1])
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        return None

    s = sample.head(FLAT_PROBE_ROWS)
    X = model._coerce(s)
    grid = np.linspace(lo, hi, FLAT_PROBE_POINTS)
    rep = X.loc[X.index.repeat(FLAT_PROBE_POINTS)].reset_index(drop=True)
    tiled = np.tile(grid, len(X))
    rep[PRICE_FEATURE] = tiled
    if BELOW_COST in rep.columns:
        rep[BELOW_COST] = (tiled < 1.0).astype(float)
    med = np.median(model._calibrated(rep).reshape(len(X), FLAT_PROBE_POINTS), axis=0)

    # first step down from the left plateau, last step up from the right one
    left = np.flatnonzero(med[0] - med > FLAT_TOL)
    right = np.flatnonzero(med - med[-1] > FLAT_TOL)
    if left.size == 0 or right.size == 0:
        if logger:
            logger.log("Price response is FLAT across all observed pricing — this model "
                       "cannot score price at all. Band suppressed; treat the win "
                       "probability as price-independent.")
        return None
    r_lo, r_hi = float(grid[left[0]]), float(grid[right[-1]])
    if r_hi <= r_lo:
        return None
    if logger:
        dead_lo = (r_lo - lo) / (hi - lo)
        logger.log(f"Price response is informative on {PRICE_FEATURE} in "
                   f"[{r_lo:.4f}, {r_hi:.4f}] — observed pricing runs [{lo:.4f}, {hi:.4f}], "
                   f"so the bottom {dead_lo:.0%} of it is a DEAD ZONE where the model "
                   f"returns a constant. The band is swept over the informative range "
                   f"only, so no price is ever recommended where the model is blind.")
    return (r_lo, r_hi)


def _monotonicity_violation(model: LineWinModel, sample: pd.DataFrame, n: int = 100,
                            above_cost_only: bool = True) -> float:
    """Fraction of sampled lines whose served price response rises.

    The guarantee the panel needs is over the range reps actually trade in, so
    this measures price_ratio >= 1.0 by default. Across the below-cost boundary
    the curve MAY rise, because `below_cost` is deliberately unconstrained — a
    line priced under cost is a distress signal, and "stop selling at a loss" is
    the correct thing for the card to say. Call with above_cost_only=False to see
    the full-range figure, which is reported alongside."""
    if model.price_bounds is None or PRICE_FEATURE not in model.feature_columns or sample.empty:
        return 0.0
    s = sample.head(n).reset_index(drop=True)
    X = model._coerce(s)
    grid = np.linspace(model.price_bounds[0], model.price_bounds[1], SWEEP_POINTS)
    rep = X.loc[X.index.repeat(SWEEP_POINTS)].reset_index(drop=True)
    tiled = np.tile(grid, len(s))
    rep[PRICE_FEATURE] = tiled
    if BELOW_COST in rep.columns:
        rep[BELOW_COST] = (tiled < 1.0).astype(float)
    p = model._calibrated(rep).reshape(len(s), SWEEP_POINTS)
    if above_cost_only:
        keep = grid >= 1.0
        if keep.sum() < 2:
            return 0.0
        p = p[:, keep]
    return float((np.diff(p, axis=1) > 1e-9).any(axis=1).mean())


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
