# Evaluation & Configuration Report — `m1_quote_line_win`

**Endpoint** `POST /api/metal_final/models/m1_quote_line_win/predict-raw`
**Serving name** `t_metal_final__m_m1_quote_line_win`
**Champion** v11 (registered 2026-08-12 15:00Z) · previous alias v9
**Report date** 2026-08-14

All figures below were read from the MLflow registry and computed by loading the
served champion artefact. Nothing is reproduced from memory.

---

## 1. Headline

The model discriminates well and is well calibrated. AUC 0.797 against a 0.527
majority-class baseline; ECE 0.027 against a 0.05 tolerance. The monotone price
guarantee holds on every held-out line.

The material finding is in the SHAP analysis: **`price_ratio` is now the dominant
feature at 45.7% of total attribution**, up from 4th place in the earlier audit. The
panel is responding to price, which is the whole point of the product and was
previously not true.

One operational note: **the champion in service was trained before early stopping was
added.** A retrain on the same data measured AUC 0.7997 / Brier 0.1831 / accuracy
0.7297 — better on every metric. That gain is currently unbanked.

---

## 2. Evaluation

Temporal-grouped holdout, 5,368 lines never used for fitting or calibration.

| Metric | Value | Reading |
|---|---|---|
| Accuracy | 0.7308 | against a 0.5270 baseline — **+20.4 points** |
| AUC | 0.7967 | ranks a winner above a loser ~80% of the time |
| PR-AUC | 0.7993 | holds up under class imbalance |
| Precision | 0.7244 | of lines called "win", 72% won |
| Recall | 0.7897 | of lines that won, 79% were called |
| F1 | 0.7556 | neither side is being sacrificed |
| Brier | 0.1843 | probability accuracy, lower better |
| ECE | 0.0271 | stated vs actual frequency — inside the 0.05 gate |
| Monotonicity violations | 0.0000 | price response never rises, on real holdout rows |

Fold sizes: 16,110 fit / 5,365 calibrate / 5,368 score. Split strategy
`temporal-grouped` — the holdout is forward in time, matching monthly retrain and
predict-forward.

**Balance.** Recall (0.790) exceeds precision (0.724), so the model leans slightly
toward calling wins. At a 50% cut-off it over-predicts conversion by roughly 6 points.
For a decision-support panel that is the safer direction — a rep is more likely to be
told a marginal deal is winnable than to be told to walk away from one that was.

---

## 3. Configuration

### 3.1 Registered parameters

| Parameter | Value |
|---|---|
| algorithm | `lightgbm+isotonic` |
| `n_estimators` | 200 |
| `learning_rate` | 0.03 |
| `num_leaves` | 15 |
| `min_child_samples` | 20 |
| `random_state` | 7 |

### 3.2 As-built configuration (read from the artefact)

| Property | Value |
|---|---|
| trees actually built | 200 |
| `best_iteration_` | 0 — **early stopping did not engage** |
| bootstrap replicas | 15 |
| isotonic output levels | 42 |
| features | 21 |
| `price_basis` | `listPrice` |
| observed price range | 0.6956 – 1.1200 |
| informative range | **0.7557 – 1.0882** |
| monotone constraint | `-1` on `price_ratio`, `0` on the other 20 |

`best_iteration_ = 0` with 200 trees confirms this champion predates the
early-stopping change; its metrics are identical to v5, so v11 is the same training
configuration re-registered.

### 3.3 Data quality of the training run

| | |
|---|---|
| rows in / out | 26,843 / 26,843 |
| duplicates removed | 0 |
| missing values filled | 0 |
| outliers capped | 2,987 |

No rows lost. Capping touched ~11% of values across 15 columns at the 0.5–99.5% band,
with `price_ratio` deliberately exempt.

---

## 4. Features used

21 features. Every one present in the export; none synthesised.

| Group | Features |
|---|---|
| Line | `quantity`, `unitPrice`, `price_ratio`, `leadTimeDays`, `contact_win_rate`, `salesrep_win_rate` |
| Line categorical | `productID`, `region`, `industry` |
| Quote context | `quote_total`, `product_type`, `payment_terms` |
| As-of history | `price_vs_product`, `price_vs_customer`, `product_win_rate`, `value_vs_customer`, `leadtime_vs_product`, `line_share`, `quote_month`, `customer_prior_quotes`, `customer_recency_days` |

Not present in this export: `days_to_expiry` (BRD "Expiration Date"), and `below_cost`,
which the below-cost check dropped because the data showed no distress dip.

---

## 5. SHAP analysis

TreeSHAP contributions to the log-odds, computed on 5,368 lines (scoring-fold size).
Base value 0.1034.

| Rank | Feature | mean \|SHAP\| | Share | Direction as the value rises |
|---:|---|---:|---:|---|
| 1 | **`price_ratio`** | **0.8165** | **45.7%** | lowers odds (corr −0.99) |
| 2 | `productID` | 0.2025 | 11.3% | categorical |
| 3 | `price_vs_product` | 0.1418 | 7.9% | lowers odds (−0.92) |
| 4 | `price_vs_customer` | 0.1208 | 6.8% | lowers odds (−0.85) |
| 5 | `salesrep_win_rate` | 0.1080 | 6.0% | **raises** odds (+0.86) |
| 6 | `quote_total` | 0.0619 | 3.5% | lowers odds (−0.64) |
| 7 | `product_win_rate` | 0.0585 | 3.3% | lowers odds (−0.72) |
| 8 | `quote_month` | 0.0556 | 3.1% | lowers odds (−0.60) |
| 9 | `customer_recency_days` | 0.0465 | 2.6% | lowers odds (−0.23) |
| 10 | `leadTimeDays` | 0.0391 | 2.2% | lowers odds (−0.23) |
| 11 | `value_vs_customer` | 0.0338 | 1.9% | ~neutral (+0.03) |
| 12 | `customer_prior_quotes` | 0.0307 | 1.7% | |
| 13 | `contact_win_rate` | 0.0275 | 1.5% | |
| 14 | `region` | 0.0192 | 1.1% | categorical |
| 15 | `payment_terms` | 0.0122 | 0.7% | categorical |
| 16 | `leadtime_vs_product` | 0.0060 | 0.3% | |
| 17 | `quantity` | 0.0026 | 0.1% | |
| 18 | `line_share` | 0.0024 | 0.1% | |
| 19 | `product_type` | 0.0022 | 0.1% | categorical |
| 20 | `industry` | 0.0004 | 0.0% | categorical |
| 21 | `unitPrice` | 0.0001 | 0.0% | |

### What it says

**Price dominates, as it should.** 45.7% of total attribution, three times the next
feature. Together the three price-relative features — `price_ratio`,
`price_vs_product`, `price_vs_customer` — carry **60.4%**. This is the finding that
changed since the earlier audit, where `price_ratio` sat 4th and the panel could not
honestly claim to be a pricing tool.

**Every price signal points the same way.** All three have strongly negative
correlation with their own SHAP: dearer lowers the odds. The `price_ratio` figure of
−0.987 is the monotone constraint holding in the attribution, not merely in the output.

**`salesrep_win_rate` is the only strong positive** at 6.0%. A rep with a better track
record raises the line's odds. Note this is an as-of feature, so it is that rep's
history *before* this quote — not a circular restatement of the outcome.

**Two attributions worth questioning:**

`productID` at 11.3% is a high-cardinality identifier. Some of that is genuine product
effect, some is memorisation. Dropping raw IDs in favour of their shrunk priors gained
AUC in the price model; the same experiment has not been run here and is worth doing.

`product_win_rate` is **negative** (−0.72) — products that historically win more are
being scored *down*. That is counter-intuitive and most likely a confound: high-win-rate
products are the commoditised ones quoted at thin margins, so the feature is partly
proxying for competitive pressure. Not necessarily wrong, but it is not the
straightforward reading, and it is exactly the kind of relationship worth checking
before anyone explains it to a customer.

**The bottom five are inert.** `quantity`, `line_share`, `product_type`, `industry` and
`unitPrice` together account for 0.3%. `unitPrice` at 0.0001 is effectively unused — the
model works entirely in ratios, which is correct, but the raw cost column is dead
weight. Removing the bottom group would simplify the model with no measurable loss.

---

## 6. Control objective

Two distinct control mechanisms. Neither is learned.

### 6.1 Monotone price constraint

`monotone_constraints = -1` on `price_ratio`, `0` on the other 20 features. The served
win probability may never rise as price rises.

This exists because the raw relationship in this data runs the wrong way — reps
discount hardest on troubled deals, so deep discounts correlate with losses. Without
the constraint the panel would tell a rep that discounting reduces their chances.

Verified on the served champion, median response across 400 lines:

| price_ratio | p(win) | |
|---:|---:|---|
| 0.6956 | 89.4% | dead zone |
| 0.7310 | 89.4% | dead zone |
| 0.7663 | 85.0% | |
| 0.8017 | 71.5% | |
| 0.8371 | 66.3% | |
| 0.8724 | 58.1% | |
| 0.9078 | 56.0% | |
| 0.9432 | 45.8% | |
| 0.9785 | 39.5% | |
| 1.0139 | 25.5% | |
| 1.0493 | 24.6% | |
| 1.0846 | 18.8% | |
| 1.1200 | 7.7% | dead zone |

Non-increasing across the whole sweep: **True**. Holdout violation rate: **0.0000**.

The response is steep and usable — 89% at 77% of list falling to 25% at list. That is a
model with real price sensitivity, not the flat curve found in the earlier audit.

### 6.2 Band objective

`band_objective: "max_win_probability_above_cost"`.

```
floor  = max(cost / basis, 0.7557)          # profitability AND evidence
p_max  = p(win) at that floor               # best odds available
band   = prices up from the floor while p(win) >= p_max - 0.05
mid    = the dearest price still within 0.025 of p_max
```

Then clamped to ±30% of the base sales price, with the cost floor taking precedence
where the two conflict. The expected-margin optimum is computed and returned as
`ev_price_*` for comparison but is **not** the recommendation.

### 6.3 Evidence boundary

Observed pricing runs 0.6956–1.1200; the response only moves between **0.7557 and
1.0882**. The bottom 14% and top 3% of observed pricing are a dead zone where the model
returns a constant. Both the sweep and the out-of-range guardrail use the informative
range, so no price is recommended and no probability is displayed where the model is
blind.

### 6.4 Display guardrails

| Rule | Threshold |
|---|---|
| hide the score | fewer than 5 comparable quotations |
| low-confidence flag | probability below 40% or above 95% |
| band tolerance | 5 percentage points |
| probability clip | 0.01 – 0.99 |

---

## 7. Recommendations

**Retrain and republish.** The served champion predates early stopping. A retrain on
the same data measured AUC 0.7997 (from 0.7967), Brier 0.1831 (from 0.1843) and
accuracy 0.7297. It should clear the promotion gate without a force publish.

**Test dropping `productID` as a raw categorical.** 11.3% attribution on a
high-cardinality identifier is partly memorisation. The equivalent change gained AUC in
the price model.

**Investigate the negative `product_win_rate`.** The sign is counter-intuitive and
likely a confound with competitive pressure. It needs an explanation before anyone
presents this model externally.

**Consider pruning the inert tail.** Five features contribute 0.3% between them.

**Standing caveat.** Price is not randomly assigned in this data, so these are
observational relationships. The monotone constraint suppresses the wrong-sign
symptom; it does not identify the causal effect of price. Any claim that "quoting at X
gives you Y%" is a statement about comparable historical quotes, not a controlled
prediction.
