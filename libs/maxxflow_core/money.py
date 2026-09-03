"""Decimal money/margin math (plan §1a mixed-precision row, §12a #6).

Schema precisions are mixed: line qty/price ``Decimal(18,4)``, totals
``Decimal(18,2)``, Product price/cost ``Decimal(12,2)``, BOM/MO qty
``Decimal(12,4)``, Item qty ``Decimal(10,2)``. Doing margin / ±30% clamp /
110%-overrun math in ``float`` can flip a Win/Loss label or a guardrail right at
the boundary, so EVERY such computation goes through ``decimal.Decimal`` here.

Also: the cached ``QuotationLineItem.lineAmount`` can disagree with a recomputed
``quantity * salesPrice``. We recompute; we do not trust the cache.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, getcontext
from typing import Iterable

getcontext().prec = 38  # ample headroom for 18,4 * 18,4 products

# Quantizers matching schema scales.
Q2 = Decimal("0.01")   # Decimal(_,2): totals, Product price/cost, Item qty
Q4 = Decimal("0.0001")  # Decimal(_,4): line qty/price, BOM/MO qty

# Guardrail constants (plan §4).
PRICE_CLAMP_PCT = Decimal("0.30")     # M1 ±30% of base Product.salesPrice
OVERRUN_THRESHOLD = Decimal("1.10")   # M3 delay label: total > 110% of expected


def D(value) -> Decimal:
    """Safe Decimal coercion. Floats go via ``str`` to avoid binary artefacts."""
    if isinstance(value, Decimal):
        return value
    if value is None:
        return Decimal("0")
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(str(value))


def q2(value) -> Decimal:
    return D(value).quantize(Q2, rounding=ROUND_HALF_UP)


def q4(value) -> Decimal:
    return D(value).quantize(Q4, rounding=ROUND_HALF_UP)


def line_amount(quantity, sales_price) -> Decimal:
    """Recompute quantity * salesPrice at Decimal(18,2) — never trust the cache."""
    return q2(D(quantity) * D(sales_price))


def gross_margin_fraction(sales_price, unit_cost) -> Decimal | None:
    """(salesPrice - unitCost) / salesPrice as a Decimal fraction. None if price<=0."""
    sp = D(sales_price)
    if sp <= 0:
        return None
    return ((sp - D(unit_cost)) / sp)


def gross_margin_amount(sales_price, unit_cost, quantity=1) -> Decimal:
    return q2((D(sales_price) - D(unit_cost)) * D(quantity))


def clamp_price(price, base_price, pct: Decimal = PRICE_CLAMP_PCT) -> Decimal:
    """Clamp a recommended price to ±pct of the base Product.salesPrice (Decimal)."""
    base = D(base_price)
    lo = base * (Decimal("1") - pct)
    hi = base * (Decimal("1") + pct)
    p = D(price)
    if p < lo:
        return q2(lo)
    if p > hi:
        return q2(hi)
    return q2(p)


def is_price_clamped(price, base_price, pct: Decimal = PRICE_CLAMP_PCT) -> bool:
    base = D(base_price)
    lo = base * (Decimal("1") - pct)
    hi = base * (Decimal("1") + pct)
    p = D(price)
    return p < lo or p > hi


def ratio(actual, expected) -> Decimal | None:
    """actual / expected as Decimal. None when expected <= 0 (0-guard)."""
    exp = D(expected)
    if exp <= 0:
        return None
    return D(actual) / exp


def is_overrun(actual, expected, threshold: Decimal = OVERRUN_THRESHOLD) -> bool:
    """M3 label: actual total duration > 110% of expected (strict, Decimal)."""
    r = ratio(actual, expected)
    if r is None:
        return False
    return r > threshold


def quantile(values: Iterable, q: float) -> Decimal | None:
    """Decimal-safe empirical quantile (linear interp) for M1 price bands."""
    xs = sorted(D(v) for v in values)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = D(str(q)) * (Decimal(len(xs) - 1))
    lo_i = int(pos)
    frac = pos - Decimal(lo_i)
    hi_i = min(lo_i + 1, len(xs) - 1)
    return xs[lo_i] + (xs[hi_i] - xs[lo_i]) * frac
