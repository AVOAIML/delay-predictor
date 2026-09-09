"""Champion/challenger promotion gate (plan §6). Auto-promote (move the
``@champion`` alias — NO redeploy) only if the challenger beats champion on the
holdout, passes calibration (Brier not materially worse, ECE <= tau) and had no
data-validation failures. Otherwise keep champion and alert. Rollback = alias
move to ``previous``.

WHY THE REASONS ARE ORDERED
---------------------------
The gate is a conjunction, so when it declines there is exactly one thing worth
saying: which check failed. The previous version appended reasons in evaluation
order and the UI showed ``reasons[0]``, which produced the reading

    "Not published — kept current champion. AUC 0.797 >= champion 0.796"

— a refusal justified by a check that PASSED. The failing check (Brier) was in
``reasons[1]`` and never surfaced, so nobody could act on it. Now every gate
reports its own verdict, blockers are listed FIRST, and ``decision.blocker``
names the single deciding one so a caller cannot accidentally show a passing
check as the reason for a refusal.

ON THE BRIER TOLERANCE
----------------------
Brier is a noisy statistic on a finite holdout, and the two numbers being
compared are not even measured on the same rows — the champion's Brier came from
ITS holdout, the challenger's from a fresh split. Demanding the third decimal
never move blocks genuinely better models over sampling noise. So Brier may
worsen by up to ``BRIER_TOL_REL`` relative, and only when AUC has STRICTLY
improved — a challenger that merely ties on AUC still has to hold calibration
exactly. Ranking and calibration cannot both drift the wrong way.

THE ABSOLUTE FLOOR
------------------
The comparison gates answer "is this better than what we serve today". They say
nothing when there is nothing to compare against, and the old order returned on
``champion is None`` BEFORE the calibration check ran. On an empty registry every
model is a first candidate, so no model was ever checked at all: a model with no
skill whatsoever would be promoted and served.

``evaluate_quality_floor`` closes that. Three checks, all against the candidate's
own held-out metrics, no champion required:

* calibration — ECE <= ``ECE_ABS_MAX``. The panel prints the score as a
  percentage, so a miscalibrated model is one that lies in the unit the user
  reads. Enforced only above ``ECE_FLOOR_MIN_ROWS`` holdout rows, below which a
  10-bin ECE measures the holdout rather than the model.
* skill — accuracy must beat the majority-class baseline. If it does not, the
  model has learned nothing that guessing "everyone wins" would not give you.
* probability skill — Brier must beat ``p(1-p)``, the Brier of a constant
  predictor that always returns the base rate. Above that line the probability
  is noise, however good the ranking looks.

The floor runs FIRST, for first candidates and challengers alike, so the two
paths cannot drift apart. A metric the caller did not supply is recorded as not
evaluated rather than silently passing — visible in ``checks``, never a blocker.
So is a metric we have but cannot trust at this sample size; see
``ECE_FLOOR_MIN_ROWS``.

ON FORCE
--------
``force=True`` is an explicit human decision to ship a candidate even when a
performance check fails. It overrides both champion-comparison checks and the
absolute performance floor. The failed checks remain in the returned audit
trail, together with a force-override check, so the API never represents a weak
model as having passed its quality gates.

Force does not override ``data_validation_passed=False``. Invalid/unusable input
data is not a model-performance choice and can fail before a trustworthy model
artifact exists to publish.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

AUC_EPS = 1e-9
BRIER_TOL_REL = 0.01     # Brier may worsen by 1% relative, but only if AUC gains

# --- the floor's own calibration limit -------------------------------------
# Deliberately NOT ece_tau: see evaluate_quality_floor.
ECE_ABS_MAX = 0.05
# ...and only where ECE means anything. A 10-bin ECE is biased UPWARD by
# sampling noise, and on a small holdout that bias alone clears the limit.
# Simulated ECE of a PERFECTLY calibrated model (300 draws, base rate ~0.5,
# the same estimator above), against a 0.05 limit:
#
#     n_test   median ECE   share of FLAWLESS models rejected
#         56       0.1361                               100%
#        100       0.1007                                99%
#        300       0.0594                                75%
#        500       0.0451                                35%
#       1200       0.0296                                 1%
#       2000       0.0227                                 0%
#       5000       0.0151                                 0%
#
# The median tracks 1/sqrt(n_test) almost exactly. Below ~1200 rows the check
# measures the holdout, not the model, so it is reported UNMEASURABLE rather
# than failed — even though a user can explicitly force a publish, the normal
# gate must not create a false failure from measurement noise.
#
# The other two floor checks need no such guard: both compare the model against
# a baseline computed on the SAME rows, so the noise largely cancels and they
# stay meaningful on small holdouts.
ECE_FLOOR_MIN_ROWS = 1200

# Every metric the gate reads. Four publish() sites had this tuple inline and
# identical, which is the shape of the original bug: the list was edited by hand
# per caller, so a key the floor needed could be — and was — dropped on the way
# in. One tuple, one helper, and a new gate metric is a one-line change here
# instead of four edits and a missed one.
GATE_METRIC_KEYS = ("auc", "brier", "ece", "accuracy", "base_rate_accuracy",
                    "accuracy_over_base_rate", "positive_rate", "n_test")


def gate_metrics(metrics: dict, **overrides) -> dict:
    """Project a full training-metrics dict onto the keys the gate reads.

    A key the trainer does not produce is simply absent, and the matching floor
    check records itself as not evaluated. Pass ``overrides`` where a trainer's
    metric means something different from the gate's reading of the name — MIL,
    for example, scores at QUOTE level, so its ``n_test`` (a line count) is the
    wrong denominator and ``n_test=metrics["n_test_quotes"]`` is the right one."""
    out = {k: metrics[k] for k in GATE_METRIC_KEYS if metrics.get(k) is not None}
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


def expected_calibration_error(y_true, p_pred, bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(p)) * abs(y_true[m].mean() - p[m].mean())
    return float(ece)


@dataclass
class PromotionDecision:
    promote: bool
    reasons: list[str]
    # The single deciding check when promotion is refused. None when promoted.
    # Show THIS next to a refusal — never reasons[0] blindly, which may be a
    # check that passed.
    blocker: str | None = None
    checks: list[dict] = field(default_factory=list)   # per-gate detail, for logs/UI

    @property
    def summary(self) -> str:
        """One line that is always true, whichever way the decision went."""
        if self.promote:
            return "; ".join(self.reasons) if self.reasons else "all gates passed"
        return self.blocker or (self.reasons[0] if self.reasons else "blocked")


def _fmt(v: float) -> str:
    """Enough digits to see a difference that the gate can act on.

    "AUC 0.7967 < champion 0.7967 (-0.0000)" reads as a tie being rejected, which
    looks like a bug. The gap was real — just below the 4th decimal and above the
    1e-9 epsilon. Print small numbers in scientific notation so the message is
    honest about how small the difference actually is."""
    a = abs(v)
    if a == 0.0:
        return "0"
    return f"{v:+.1e}" if a < 5e-5 else f"{v:+.4f}"


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": passed, "detail": detail}


def _skipped(name: str, metric: str) -> dict:
    """A check the caller gave us no input for. Recorded so it is visible in the
    UI and the logs, but it passes — an absent metric is a plumbing gap, not
    evidence the model is bad, and failing on it would block every trainer that
    has not been taught to forward the key yet."""
    return {"name": name, "passed": True, "skipped": True,
            "detail": f"{name}: not evaluated — no {metric!r} in the candidate metrics"}


def _unmeasurable(name: str, why: str) -> dict:
    """A check we HAVE the input for but cannot trust on this much data. Distinct
    from ``_skipped``: nothing is missing, the statistic is simply too noisy at
    this sample size to say anything. Passes for the same reason ``_skipped``
    does — refusing on a number we have just admitted is unreliable would be the
    same mistake in the other direction — but says so where anyone can read it."""
    return {"name": name, "passed": True, "skipped": True, "unmeasurable": True,
            "detail": f"{name}: not evaluated — {why}"}


def evaluate_quality_floor(candidate: dict, *, ece_tau: float = 0.05) -> list[dict]:
    """Is this model worth serving AT ALL — independent of any champion.

    Returns the per-check records; the caller turns them into a decision. Every
    threshold here is absolute, so the same list is produced for a first
    candidate and for a challenger (see the module docstring)."""
    checks: list[dict] = []

    # --- calibration: does the printed percentage mean what it says -----------
    # ECE_ABS_MAX, not ece_tau. ece_tau is the CHALLENGER comparison's threshold
    # and a caller may loosen it for a particular model; the floor is the
    # platform's own limit and must not follow it down. Passing ece_tau=0.9 used
    # to switch the floor's calibration check off along with the comparison one.
    ece = candidate.get("ece")
    n_test = candidate.get("n_test")
    if ece is None:
        checks.append(_skipped("calibration", "ece"))
    elif n_test is None:
        checks.append(_unmeasurable(
            "calibration",
            f"no 'n_test' in the candidate metrics, so the observed ECE "
            f"{float(ece):.4f} cannot be separated from small-holdout sampling noise"))
    elif int(n_test) < ECE_FLOOR_MIN_ROWS:
        checks.append(_unmeasurable(
            "calibration",
            f"a {int(n_test)}-row holdout is below {ECE_FLOOR_MIN_ROWS}, where a 10-bin ECE "
            f"is dominated by sampling noise (a PERFECTLY calibrated model scores about "
            f"{1.0 / max(int(n_test), 1) ** 0.5:.3f} here on noise alone). Observed ECE "
            f"{float(ece):.4f} is reported, not enforced"))
    else:
        ece = float(ece)
        ok = ece <= ECE_ABS_MAX
        checks.append(_check("calibration", ok,
                             f"ECE {ece:.4f} <= {ECE_ABS_MAX} on {int(n_test)} holdout rows"
                             if ok else
                             f"ECE {ece:.4f} > {ECE_ABS_MAX} on {int(n_test)} holdout rows — the "
                             f"score is printed as a percentage, so a miscalibrated model "
                             f"misstates it in the one unit the user reads"))

    # --- skill: does it beat guessing the majority class ---------------------
    over_base = candidate.get("accuracy_over_base_rate")
    if over_base is None:
        checks.append(_skipped("skill", "accuracy_over_base_rate"))
    else:
        over_base = float(over_base)
        ok = over_base > 0.0
        acc, base = candidate.get("accuracy"), candidate.get("base_rate_accuracy")
        where = (f" (accuracy {float(acc):.4f} vs baseline {float(base):.4f})"
                 if acc is not None and base is not None else "")
        checks.append(_check("skill", ok,
                             f"accuracy beats the majority-class baseline by "
                             f"{_fmt(over_base)}{where}" if ok else
                             f"accuracy does NOT beat the majority-class baseline "
                             f"({_fmt(over_base)}){where} — the model has learned nothing "
                             f"that always guessing the commoner outcome would not give"))

    # --- probability skill: does the probability carry information -----------
    # p(1-p) is the Brier score of a constant predictor that always returns the
    # base rate. A model above that line is worse than saying nothing at all.
    brier, rate = candidate.get("brier"), candidate.get("positive_rate")
    if brier is None or rate is None:
        checks.append(_skipped("probability_skill",
                               "brier" if brier is None else "positive_rate"))
    else:
        brier, rate = float(brier), float(rate)
        zero_skill = rate * (1.0 - rate)
        if zero_skill <= 0.0:
            checks.append(_check("probability_skill", False,
                                 f"positive rate {rate:.4f} is degenerate — the holdout has "
                                 f"only one class, so nothing was actually measured"))
        else:
            ok = brier < zero_skill
            checks.append(_check("probability_skill", ok,
                                 f"Brier {brier:.4f} < {zero_skill:.4f} from a constant "
                                 f"predictor at the {rate:.4f} base rate" if ok else
                                 f"Brier {brier:.4f} >= {zero_skill:.4f}, the score a constant "
                                 f"predictor at the {rate:.4f} base rate would get — the "
                                 f"probability carries no information"))
    # Tag floor checks so the returned audit trail distinguishes absolute
    # performance failures from champion-comparison failures.
    for c in checks:
        c["floor"] = True
    return checks


def _verdict(checks: list[dict], *, force: bool) -> PromotionDecision:
    """Turn a list of checks into a decision. The ONLY place ordering, blocker
    selection and the force override are decided, so the first-candidate and
    challenger paths cannot drift apart."""
    failed = [c for c in checks if not c["passed"]]
    passed = [c for c in checks if c["passed"]]
    # blockers first, so reasons[0] is never a passing check next to a refusal
    reasons = [c["detail"] for c in failed] + [c["detail"] for c in passed]
    if not failed:
        return PromotionDecision(True, reasons, checks=checks)
    if force:
        # Preserve every failed performance check and add an explicit audit
        # record showing that a human chose to override them.
        detail = (
            "forced publish — user overrode failed performance gate(s): "
            + ", ".join(c["name"] for c in failed)
        )
        note = _check("force", True, detail)
        return PromotionDecision(True, [detail] + reasons, blocker=None,
                                 checks=checks + [note])
    return PromotionDecision(False, reasons, blocker=failed[0]["detail"], checks=checks)


def should_promote(challenger: dict, champion: dict | None, *, ece_tau: float = 0.05,
                   data_validation_passed: bool = True, force: bool = False) -> PromotionDecision:
    if not data_validation_passed:
        r = "data-validation failed — never promote"
        return PromotionDecision(False, [r], blocker=r,
                                 checks=[_check("data_validation", False, r)])
    # The absolute floor runs FIRST, and for both paths. An empty registry used to
    # return here before any check ran, so nothing was ever verified.
    checks = evaluate_quality_floor(challenger, ece_tau=ece_tau)

    if champion is None:
        # No incumbent: the comparison gates have nothing to compare against, so
        # clearing the floor is the whole test. Something still has to be
        # deployed, and `force` remains available for a deliberate demo publish
        # of a model that does not clear it.
        checks.append(_check("incumbent", True,
                             "no incumbent champion — no comparison to make, so the "
                             "absolute floor is the whole gate"))
        return _verdict(checks, force=force)

    c_auc, c_brier = float(challenger["auc"]), float(challenger["brier"])
    m_auc, m_brier = float(champion["auc"]), float(champion["brier"])

    # --- ranking ------------------------------------------------------------
    auc_gain = c_auc - m_auc
    auc_ok = auc_gain >= -AUC_EPS
    auc_strict = auc_gain > AUC_EPS
    checks.append(_check("auc", auc_ok,
                         f"AUC {c_auc:.6f} {'>=' if auc_ok else '<'} champion {m_auc:.6f}"
                         f" ({_fmt(auc_gain)})"))

    # --- calibration vs champion -------------------------------------------
    # Tolerance only where a real ranking gain pays for it (see module docstring).
    allowance = m_brier * BRIER_TOL_REL if auc_strict else 1e-6
    brier_ok = c_brier <= m_brier + allowance
    if brier_ok:
        detail = (f"Brier {c_brier:.4f} within tolerance of champion {m_brier:.4f}"
                  + (f" (+{allowance:.4f} allowed, AUC improved)" if auc_strict else ""))
    else:
        detail = (f"Brier {c_brier:.6f} WORSE than champion {m_brier:.6f} "
                  f"by {_fmt(c_brier - m_brier)}"
                  + (f", over the {allowance:.4f} allowed for an AUC gain" if auc_strict
                     else " — and AUC did not improve, so no tolerance applies"))
    checks.append(_check("brier", brier_ok, detail))

    return _verdict(checks, force=force)
