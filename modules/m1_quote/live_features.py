"""Turn ONE raw, business-facing quote (header + product lines) into the engineered
feature row(s) m1_quote_win / m1_quote_price expect.

This is the live-serving counterpart to modules/m1_quote/raw_ingest.py's bulk
aggregation: same math, applied to a single ad-hoc quote instead of a whole training
file. It exists so the Test Predictions UI can show a client raw columns only
(customerID, quantity, unitPrice, ...) and never a computed ratio or win-rate — the
client changes raw values, this module derives what the model actually consumes.

Keep the formulas here in sync with raw_ingest.py if the derivation changes."""

from __future__ import annotations

from m1_quote.raw_ingest import price_ratio_for

DEFAULT_LIST_PRICE_MARKUP = 0.67
_PAYMENT_TERM_MODEL_VALUES = {
    "immediate payment": "Immediate",
    "immediate": "Immediate",
    "15 days": "15 Days",
    "21 days": "21 Days",
}


def _rate(lookup: dict, key) -> float:
    """Look up a customer/rep's historical win-rate; unknown/blank key -> global rate."""
    if key in (None, ""):
        return float(lookup["_global"])
    return float(lookup.get(str(key), lookup["_global"]))


def quote_features_from_raw(header: dict, lines: list[dict], rate_lookup: dict) -> dict:
    """header: customerID, salesRepID, region, industry, leadTimeDays (all raw, optional
    except region/industry which the model needs as categories).
    lines: [{productID, quantity, unitPrice, salesPrice, negotiatedSalesPrice?}, ...]
    rate_lookup: {"contact_win_rate": {customerID: rate, "_global": rate},
                  "salesrep_win_rate": {salesRepID: rate, "_global": rate}}
    (see data_creation_script.py's _entity_rate_lookup for how this is built)."""
    if not lines:
        raise ValueError("at least one product line is required")

    ratios: list[float] = []
    amounts: list[float] = []
    qty_total = 0.0
    product_ids = set()

    for i, ln in enumerate(lines):
        try:
            unit_price = float(ln["unitPrice"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"line {i}: unitPrice is required and must be numeric")
        negotiated = ln.get("negotiatedSalesPrice")
        effective = float(negotiated) if negotiated not in (None, "") else float(ln.get("salesPrice", unit_price))
        qty = float(ln.get("quantity", 1) or 1)
        ratio = (effective / unit_price) if unit_price else 1.0
        ratios.append(ratio)
        amounts.append(qty * effective)
        qty_total += qty
        product_ids.add(ln.get("productID", i))

    mean_ratio = sum(ratios) / len(ratios)
    min_ratio = min(ratios)
    amt_sum = sum(amounts)
    wtd_ratio = (sum(r * a for r, a in zip(ratios, amounts)) / amt_sum) if amt_sum > 0 else mean_ratio

    return {
        "grand_total": amt_sum,
        "total_quantity": qty_total,
        "line_count": len(lines),
        "n_products": len(product_ids),
        "wtd_price_ratio": wtd_ratio,
        "mean_price_ratio": mean_ratio,
        "min_price_ratio": min_ratio,
        "avg_discount_pct": (1.0 - mean_ratio) * 100.0,
        "contact_win_rate": _rate(rate_lookup["contact_win_rate"], header.get("customerID")),
        "salesrep_win_rate": _rate(rate_lookup["salesrep_win_rate"], header.get("salesRepID")),
        "region": header.get("region", ""),
        "industry": header.get("industry", ""),
    }


def _first_positive(*values) -> float | None:
    """First value that parses as a price greater than zero. A blank, a null or a
    zero is 'not supplied', never 'free' — see line_win_features_from_raw."""
    for v in values:
        if v in (None, ""):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f > 0:
            return f
    return None


def _payment_term_for_model(value):
    """Translate fixed UI labels to the category spelling used during training."""
    if value in (None, ""):
        return None
    text = str(value).strip()
    return _PAYMENT_TERM_MODEL_VALUES.get(text.lower(), text)


def line_win_features_from_raw(header: dict, lines: list[dict], rate_lookup: dict) -> list[dict]:
    """m1_quote_line_win predicts a win probability PER LINE ITEM (one feature row
    per line, like price_features_from_raw) — but unlike the price model, the
    ratio here is the line's OWN proposed price_ratio (effective_price/listPrice),
    the same signal quote_features_from_raw aggregates across the whole quote,
    just kept per-line instead. header: customerID, salesRepID, region, industry,
    leadTimeDays, paymentTerms?. lines: [{productID, quantity, unitPrice,
    salesPrice, negotiatedSalesPrice?, materialSpec?}, ...].

    The optional keys carry the remaining BRD "Training Data Elements" that
    raw_ingest.build_line_frame now supplies at TRAINING time (materialSpec ->
    product_type, paymentTerms -> payment_terms); quote_total is summed from the
    submitted lines here, exactly as ingest sums it from the export.
    They are emitted here under the SAME names so a model trained with them is
    served with them; a champion trained without them simply ignores the extra
    keys, so callers never have to know which build they are talking to."""
    if not lines:
        raise ValueError("at least one product line is required")

    contact_rate = _rate(rate_lookup["contact_win_rate"], header.get("customerID"))
    salesrep_rate = _rate(rate_lookup["salesrep_win_rate"], header.get("salesRepID"))
    records, effective_prices, quantities = [], [], []
    for i, ln in enumerate(lines):
        try:
            unit_price = float(ln["unitPrice"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"line {i}: unitPrice is required and must be numeric")
        effective = _first_positive(ln.get("negotiatedSalesPrice"), ln.get("salesPrice"))
        if effective is None:
            effective = unit_price
        list_price = _first_positive(ln.get("listPrice"))
        if list_price is None:
            list_price = unit_price + (DEFAULT_LIST_PRICE_MARKUP * unit_price)
        # Same basis as training: the ratio and the price-band basis must use the
        # identical list price, whether supplied or derived.
        ratio, _basis = price_ratio_for(effective, list_price, unit_price)
        effective_prices.append(effective)
        rec = {
            "productID": ln.get("productID", ""),
            "quantity": float(ln.get("quantity", 1) or 1),
            "unitPrice": unit_price,
            "price_ratio": ratio,
            "leadTimeDays": float(header.get("leadTimeDays", 0) or 0),
            "region": header.get("region", ""),
            "industry": header.get("industry", ""),
            "contact_win_rate": contact_rate,
            "salesrep_win_rate": salesrep_rate,
            "list_price": list_price,
        }
        # optional, only when the caller supplied them — an absent key is filled
        # with the TRAINING median/unknown level by LineWinModel._coerce, which is
        # a better default than a zero invented here
        if ln.get("materialSpec") not in (None, ""):
            rec["product_type"] = ln["materialSpec"]
        payment_term = _payment_term_for_model(header.get("paymentTerms"))
        if payment_term is not None:
            rec["payment_terms"] = payment_term
        quantities.append(rec["quantity"])
        records.append(rec)

    # quote_total is DERIVED from the submitted lines with the same formula
    # raw_ingest.build_line_frame uses at training time, so the caller never has
    # to supply it and the two can never drift. Every line of a quote carries the
    # same total, exactly as in training.
    total = float(sum(q * p for q, p in zip(quantities, effective_prices)))
    for rec in records:
        rec["quote_total"] = total
    return records


def mil_features_from_raw(header: dict, lines: list[dict]) -> list[dict]:
    """m1_quote_mil (Noisy-OR MIL) predicts a per-line win probability from ALL
    of a quote's lines scored together as one BAG — so every returned record
    shares the same `quotationID` (a fixed placeholder), which is exactly how
    MILModel.predict groups rows into a bag for its Noisy-OR aggregation.
    Unlike the other *_features_from_raw functions, no rate_lookup is needed —
    this model doesn't use contact/salesrep historical win-rates as features.
    header: salesRepID, region, industry, leadTimeDays. lines: [{productID,
    quantity, unitPrice, salesPrice, materialSpec, discountPercent}, ...]."""
    if not lines:
        raise ValueError("at least one product line is required")

    records = []
    for i, ln in enumerate(lines):
        try:
            unit_price = float(ln["unitPrice"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"line {i}: unitPrice is required and must be numeric")
        try:
            sales_price = float(ln["salesPrice"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"line {i}: salesPrice is required and must be numeric")
        records.append({
            "quotationID": "DRAFT-PREVIEW",
            "productID": ln.get("productID", ""),
            "quantity": float(ln.get("quantity", 1) or 1),
            "unitPrice": unit_price,
            "salesPrice": sales_price,
            "materialSpec": ln.get("materialSpec", "Other"),
            "discountPercent": float(ln.get("discountPercent", 0) or 0),
            "leadTimeDays": float(header.get("leadTimeDays", 0) or 0),
            "region": header.get("region", ""),
            "industry": header.get("industry", ""),
            "salesRepID": header.get("salesRepID", ""),
        })
    return records


def price_features_from_raw(header: dict, lines: list[dict], rate_lookup: dict) -> list[dict]:
    """m1_quote_price predicts a price band PER LINE ITEM, not one aggregate for the
    whole quote (unlike quote_features_from_raw) — so this returns one feature row
    per line, not a single combined one. materialSpec isn't collected from this raw
    form (see raw_ingest.build_price_frame's docstring — it's sourced from the DB
    instead), but a champion trained via the DB path DOES have it in its logged
    MLflow signature, and signature enforcement happens INSIDE model.predict()
    before the adaptive feature-selection code ever runs — so it must always be
    present in the record (defaulted to "unknown"), or serving raises a hard
    MlflowException instead of gracefully ignoring the unused feature.
    header: customerID, region, leadTimeDays. lines: [{productID, quantity, unitPrice}, ...]."""
    if not lines:
        raise ValueError("at least one product line is required")

    contact_rate = _rate(rate_lookup["contact_win_rate"], header.get("customerID"))
    records = []
    for i, ln in enumerate(lines):
        try:
            unit_price = float(ln["unitPrice"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"line {i}: unitPrice is required and must be numeric")
        records.append({
            "quantity": float(ln.get("quantity", 1) or 1),
            "unitPrice": unit_price,
            "leadTimeDays": float(header.get("leadTimeDays", 0) or 0),
            "contact_win_rate": contact_rate,
            "productID": ln.get("productID", ""),
            "materialSpec": ln.get("materialSpec", "unknown"),
            "region": header.get("region", ""),
        })
    return records
