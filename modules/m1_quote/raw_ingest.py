"""Build a gold_quote_win.csv-shaped training frame from a raw combined line-item export.

Input: one CSV, one row per quotation LINE (see column reference in the M1 retrain
docs) — quote-header fields (customerID, region, industry, quoteDate, grandTotal,
statusName, ...) repeat across every line of the same quotationID; revisions of the
same deal are linked via parentQuotationID (null on the original) + revisionNumber.

Output: one row per quote FAMILY (original + its revisions collapsed into one),
matching csv_win.py's expected columns:
    quotationID, tenant, grand_total, total_quantity, line_count, region, industry,
    lead_time_days, quoteDate, won, wtd_price_ratio, mean_price_ratio,
    min_price_ratio, avg_discount_pct, n_products, contact_win_rate, salesrep_win_rate

Design mirrors db_features.py::build_win_frame() so the CSV path and the DB path
produce the same feature shape:
  - won/is_closed derived from statusName, not from discountApprovalOutcome (that
    would leak the label).
  - Only the FINAL revision of each quote family is used — negotiation rounds are
    the same deal, not independent training examples.
  - effective_price = negotiatedSalesPrice if present else salesPrice; price ratio
    = effective_price / unitPrice folds discounting in directly, so avg_discount_pct
    is derived from the ratio rather than the raw (mostly-null) discountPercent column.
  - contact_win_rate / salesrep_win_rate are Bayesian-smoothed, AS-OF each quote's
    quoteDate (only prior-closed families count) — the #1 way to leak the label is
    to compute these over the whole dataset instead of as-of.

Used by both dataset/script/data_creation_script.py (offline CLI regeneration) and
services/configurator/jobs.py (the "Retrain with new data" upload path in the UI) —
one place for this math so the two never drift apart."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_SMOOTH = 3.0

# A quote sent to a customer and left unanswered this long is treated as LOST.
# Without it the training set only contains deals someone bothered to close, which
# skews it heavily toward wins and starves the model of the losses it most needs
# to learn. The reference "now" is the export's own newest quoteDate, never the
# wall clock, so a retrain on the same file always produces the same labels.
OPEN_QUOTE_LOST_AFTER_DAYS = 60

# --- schema: what an export MUST carry vs what it may carry -----------------
# Split deliberately. Exports differ between MaXXFlow deployments — a tenant with
# no revision workflow has no parentQuotationID/revisionNumber, and one that does
# not record a separately negotiated price has no negotiatedSalesPrice. None of
# those are needed to derive a label or a price ratio, so demanding them rejects
# a perfectly trainable file at the upload gate. Anything optional is synthesized
# to its neutral value with the fallback documented next to it.
HEADER_REQUIRED = ["quotationID", "industry", "customerID", "region", "salesRepID",
                   "quoteDate", "leadTimeDays", "statusName", "grandTotal"]
HEADER_OPTIONAL = {
    "parentQuotationID": "no revision chains — every quote is its own family",
    "revisionNumber": "assumed 1 (single revision per quote)",
    "customerName": "display only, never a feature",
    "paymentTerms": "used as a categorical feature when present",
}
LINE_REQUIRED = ["quotationID", "productID", "quantity", "unitPrice", "salesPrice"]
LINE_OPTIONAL = {
    "negotiatedSalesPrice": "falls back to salesPrice",
    "lineTotal": "falls back to quantity x effective price",
    "materialSpec": "carried through as product_type (the BRD's Product Type)",
    "discountPercent": "MIL only; the other builders derive discount from the price ratio",
    "productName": "display only",
    "quotationLineItemID": "identifier, price frame only",
}

HEADER_COLS = HEADER_REQUIRED + list(HEADER_OPTIONAL)
LINE_COLS = LINE_REQUIRED + list(LINE_OPTIONAL)
# what the upload gate actually enforces (dedup, order-stable)
RAW_REQUIRED_COLS = list(dict.fromkeys(HEADER_REQUIRED + LINE_REQUIRED))
# present -> better model; absent -> synthesized. Shown in the Review & Clean step
# so a user can see what their export is leaving on the table.
RAW_OPTIONAL_COLS = list(dict.fromkeys(list(HEADER_OPTIONAL) + list(LINE_OPTIONAL)))
# build_mil_frame() genuinely needs these — unlike the other builders, the MIL
# model engineers its own features straight from materialSpec/discountPercent
# rather than deriving a price_ratio, so they are required inputs there.
MIL_REQUIRED_COLS = list(dict.fromkeys(
    RAW_REQUIRED_COLS + ["materialSpec", "discountPercent", "productName"]))
MIL_OPTIONAL_COLS = [c for c in RAW_OPTIONAL_COLS if c not in MIL_REQUIRED_COLS]


def _fill_optional(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Add any absent optional column at its neutral value, so every builder below
    can read the full schema without a `if c in df` at each use site."""
    out = df
    missing = [c for c in spec if c not in out.columns]
    if missing:
        out = out.copy()
        for c in missing:
            out[c] = 1 if c == "revisionNumber" else np.nan
    return out


def _effective_price(li: pd.DataFrame) -> pd.Series:
    """negotiatedSalesPrice when it exists, else the quoted salesPrice. One
    definition, shared by every builder, so the price ratio cannot drift."""
    sale = pd.to_numeric(li["salesPrice"], errors="coerce")
    if "negotiatedSalesPrice" not in li.columns:
        return sale
    return pd.to_numeric(li["negotiatedSalesPrice"], errors="coerce").fillna(sale)


def _price_basis(li: pd.DataFrame) -> tuple[pd.Series, str]:
    """What to measure the quoted price AGAINST — list price if the export has it,
    otherwise unit cost.

    This choice is the single most consequential one in the whole feature set. A
    buyer never sees your cost; what they react to is the discount off LIST. A
    ratio against cost mixes the pricing decision together with per-product cost
    variation, and on a catalogue where costs move independently of list prices
    that noise can swamp the price signal completely — measured on a controlled
    fixture with a deliberate 41-point price effect, the cost-based ratio
    recovered none of it and the list-based ratio recovered nearly all."""
    if "listPrice" in li.columns:
        base = pd.to_numeric(li["listPrice"], errors="coerce").replace(0, np.nan)
        if base.notna().any():
            return base, "listPrice"
    return pd.to_numeric(li["unitPrice"], errors="coerce").replace(0, np.nan), "unitPrice"


def price_ratio_for(sales_price, list_price=None, unit_cost=None) -> tuple[float, str]:
    """Scalar twin of _price_ratio, for the SERVING path.

    Exists so live_features cannot compute the ratio a different way from
    training. It already did once: training moved to a list-price basis while
    serving stayed on cost, every incoming ratio landed above the trained range,
    and the monotone constraint pinned every line to the same floor probability.
    Two identical predictions for two different products was the only symptom.
    One function, both callers, no drift."""
    sp = float(sales_price)
    for base, name in ((list_price, "listPrice"), (unit_cost, "unitPrice")):
        try:
            b = float(base)
        except (TypeError, ValueError):
            continue
        if b > 0:
            return sp / b, name
    return 1.0, "none"


def _price_ratio(li: pd.DataFrame) -> pd.Series:
    """Quoted price as a fraction of its basis. Against list price, <1 is a
    discount and >1 a premium; against cost (no list price available), >1 sells
    above cost. Either way LOWER means cheaper, so the model's monotone
    constraint points the same way in both cases."""
    return _effective_price(li) / _price_basis(li)[0]

# where the CLI writes, and the API reads, the customer/rep win-rate lookup used by
# live predict-raw (services/configurator/app.py) — one shared path so retraining
# and live prediction never point at different files.
RATE_LOOKUP_PATH = Path(__file__).resolve().parents[2] / "dataset" / "win_rate_lookup.json"
# Sibling of RATE_LOOKUP_PATH, written by the same retrain job and read by the
# Test-Predictions schema endpoint: which productIDs actually belong to an
# industry, and which materialSpecs to a product. Same one-file-shared-by-
# training-and-serving discipline, so the dropdowns can never offer a
# combination the export has never contained.
OPTION_GRAPH_PATH = Path(__file__).resolve().parents[2] / "dataset" / "option_graph.json"

# child column -> the parent whose value narrows it. Raw export column names,
# because these drive the RAW test-input form, not the engineered feature names.
OPTION_PARENTS = {"productID": "industry", "materialSpec": "productID"}


def build_option_graph(raw: pd.DataFrame, max_children: int = 500) -> dict:
    """{child: {"parent": name, "options_by": {parent_value: [child values]}}}

    Built from the WHOLE export rather than the decided subset: a product that
    exists but has no closed quotes yet should still be selectable — the model's
    own comparables guardrail is what tells the user the evidence is thin, and
    silently hiding the product would be a worse answer than a Low Confidence
    card. Values are strings so they compare cleanly against form state."""
    graph: dict = {}
    for child, parent in OPTION_PARENTS.items():
        if child not in raw.columns or parent not in raw.columns:
            continue
        pairs = raw[[parent, child]].dropna().astype(str).drop_duplicates()
        options_by = {k: sorted(v)[:max_children]
                      for k, v in pairs.groupby(parent)[child].unique().items()}
        if options_by:
            graph[child] = {"parent": parent, "options_by": options_by}
    return graph


def _resolve_family_root(df_header: pd.DataFrame) -> dict:
    """quotationID -> root quotationID, by following parentQuotationID back to null."""
    parent = dict(zip(df_header["quotationID"], df_header["parentQuotationID"]))
    root: dict = {}

    def find(qid):
        if qid in root:
            return root[qid]
        p = parent.get(qid)
        r = qid if (p is None or (isinstance(p, float) and np.isnan(p))) else find(p)
        root[qid] = r
        return r

    for qid in parent:
        find(qid)
    return root


# An explicit 0/1 outcome column, if the export already carries one. Checked
# before statusName, because a column that literally says "did we win" beats any
# amount of string matching against a vocabulary that varies per deployment.
LABEL_COLUMNS = ["target_win", "won", "is_won", "isWon"]

# Word-boundary matches, NOT substrings. The old substring test read "won" and
# "lost", so an export using WIN / LOSS matched neither and every quote came back
# undecided — which the aging rule below then turned into an all-loss dataset.
_WIN_WORDS = r"\b(won|win|wins|confirmed|accepted|converted|order|ordered|success|successful)\b"
_LOSS_WORDS = r"\b(lost|loss|lose|rejected|declined|cancelled|canceled|failed|unsuccessful)\b"


def _label(status: pd.Series) -> pd.Series:
    s = status.astype(str).str.lower()
    won = s.str.contains(_WIN_WORDS, regex=True, na=False)
    lost = s.str.contains(_LOSS_WORDS, regex=True, na=False)
    # a status matching BOTH ("won back after loss") is ambiguous, not a win
    return np.where(won & ~lost, 1.0, np.where(lost & ~won, 0.0, np.nan))


def _derive_label(header: pd.DataFrame) -> tuple[pd.Series, str]:
    """The quote outcome, plus a one-line description of where it came from.

    Fails LOUDLY when a status vocabulary is unrecognised. Silently labelling an
    entire export as one class is the worst possible outcome: training then
    either crashes with a confusing message or, worse, succeeds on garbage."""
    for col in LABEL_COLUMNS:
        if col in header.columns:
            v = pd.to_numeric(header[col], errors="coerce")
            if v.notna().any() and set(v.dropna().unique()) <= {0.0, 1.0}:
                return v, f"explicit label column '{col}'"

    if "statusName" not in header.columns:
        raise ValueError(
            "cannot derive a win/loss label: no explicit label column "
            f"({LABEL_COLUMNS}) and no 'statusName'")

    lab = pd.Series(_label(header["statusName"]), index=header.index)
    if lab.notna().sum() == 0:
        seen = sorted(header["statusName"].astype(str).str.strip().unique())[:20]
        raise ValueError(
            "cannot derive a win/loss label: none of the statusName values look like a "
            f"win or a loss. Found: {seen}. Either add a 0/1 column named one of "
            f"{LABEL_COLUMNS}, or extend _WIN_WORDS/_LOSS_WORDS in raw_ingest.py.")
    return lab, "statusName"


def _asof_rate(df: pd.DataFrame, entity: str) -> np.ndarray:
    """Bayesian-smoothed as-of win-rate over `entity`, using only quotes whose
    quoteDate is strictly earlier than the current row's — no peeking at the future
    (or the current row's own outcome)."""
    order = df["quoteDate"].to_numpy()
    won = df["won"].to_numpy()
    closed = df["is_closed"].to_numpy()
    ent = df[entity].to_numpy()
    g = float(np.nanmean(won[closed])) if closed.any() else 0.5
    out = np.full(len(df), g)
    for i in range(len(df)):
        prior = closed & (order < order[i]) & (ent == ent[i])
        n = int(prior.sum())
        w = int(np.nansum(won[prior]))
        out[i] = (w + _SMOOTH * g) / (n + _SMOOTH)
    return out


def _entity_rate_lookup(df: pd.DataFrame, entity: str) -> dict:
    """Smoothed win-rate per entity value, using ALL of its closed-quote history —
    the live counterpart to _asof_rate: at serving time "now" is after every training
    row, so there's no future to peek at. `_global` is the fallback for an entity
    (e.g. a brand-new customer) never seen in training."""
    global_rate = float(df["won"].mean())
    g = df.groupby(entity)["won"].agg(["sum", "count"])
    rate = (g["sum"] + _SMOOTH * global_rate) / (g["count"] + _SMOOTH)
    return {"_global": global_rate, **{str(k): float(v) for k, v in rate.items()}}


def _age_out_open_quotes(header: pd.DataFrame, lost_after_days: int | None) -> pd.DataFrame:
    """Label a stale open quote as lost. Silence past a point IS an outcome.

    Two deliberate restrictions. A DRAFT is never aged out — it was never sent to
    anyone, so it cannot have been refused; it is an abandoned internal document
    and stays undecided. And "now" is the newest quoteDate in the export rather
    than today's date, so labels do not drift every time the same file is
    retrained.

    Pass lost_after_days=None to disable and keep the old drop-everything-open
    behaviour."""
    if lost_after_days is None or header.empty:
        return header
    ref = header["quoteDate"].max()
    if pd.isna(ref):
        return header
    age = (ref - header["quoteDate"]).dt.days
    never_sent = header["statusName"].astype(str).str.lower().str.contains("draft")
    aged = header["won"].isna() & ~never_sent & (age >= lost_after_days)
    header.loc[aged, "won"] = 0.0
    return header


# --- strictly-prior ("as-of") history ---------------------------------------
def _prior_cumulative(events: pd.DataFrame, by: str, value: str) -> tuple[pd.Series, pd.Series]:
    """(prior sum, prior count) of `value` within `by`, strictly BEFORE each row's
    quoteDate, aligned back to `events.index`.

    Ties on the same day are deliberately NOT counted as prior: two quotes sent
    the same morning cannot have informed each other, and for anything derived
    from `won` a same-day sibling's outcome is not knowable at quote time. This is
    the same discipline as _asof_rate, done with merge_asof so it stays O(n log n)
    as more history features are added."""
    e = events[[by, "quoteDate", value]].copy()
    e[by] = e[by].astype(str)
    e[value] = pd.to_numeric(e[value], errors="coerce")
    daily = (e.groupby([by, "quoteDate"], as_index=False)
               .agg(_n=(value, "count"), _s=(value, "sum"))
               .sort_values("quoteDate"))
    daily["_cn"] = daily.groupby(by)["_n"].cumsum()
    daily["_cs"] = daily.groupby(by)["_s"].cumsum()
    left = e[[by, "quoteDate"]].reset_index().sort_values("quoteDate")
    m = pd.merge_asof(left, daily[[by, "quoteDate", "_cn", "_cs"]],
                      on="quoteDate", by=by, allow_exact_matches=False)
    m = m.set_index("index").reindex(events.index)
    return m["_cs"], m["_cn"]


def _prior_ratio(events: pd.DataFrame, by: str, value: str) -> pd.Series:
    """This row's `value` divided by the prior average of `value` for its `by`
    group. 1.0 means "normal for this product/customer", 1.2 means "20% above what
    we usually do". A RELATIVE position, which is what a buyer actually reacts to
    — absolute margin over cost is not comparable across a catalogue where unit
    costs span two orders of magnitude. Falls back to 1.0 with no history."""
    s, n = _prior_cumulative(events, by, value)
    prior = (s / n).where(n > 0)
    cur = pd.to_numeric(events[value], errors="coerce")
    return (cur / prior).replace([np.inf, -np.inf], np.nan).fillna(1.0)


def _prior_win_rate(events: pd.DataFrame, by: str) -> pd.Series:
    """Bayesian-smoothed prior win rate within `by`, aligned to events.index.

    De-duplicated to one row per (group, quotation) before counting, so a quote
    carrying the same product on three lines contributes one observation rather
    than three — otherwise a big multi-line quote would dominate its product's
    history purely by being long."""
    uniq = events.drop_duplicates(subset=[by, "quotationID"])[[by, "quotationID", "quoteDate", "won"]]
    g = float(pd.to_numeric(uniq["won"], errors="coerce").mean())
    s, n = _prior_cumulative(uniq, by, "won")
    uniq = uniq.assign(_rate=(s.fillna(0.0) + _SMOOTH * g) / (n.fillna(0.0) + _SMOOTH))
    uniq[by] = uniq[by].astype(str)
    merged = events[[by, "quotationID"]].assign(**{by: events[by].astype(str)}).merge(
        uniq[[by, "quotationID", "_rate"]], on=[by, "quotationID"], how="left")
    return pd.Series(merged["_rate"].to_numpy(), index=events.index).fillna(g)


def _build_header(raw: pd.DataFrame,
                  lost_after_days: int | None = OPEN_QUOTE_LOST_AFTER_DAYS) -> pd.DataFrame:
    """One row per DECIDED quote family (revisions collapsed to the final one, open
    quotes dropped), with the as-of contact/salesrep win-rates already computed.
    Shared by build() (WIN) and build_price_frame() (PRICE) so both models see the
    exact same family-collapse + as-of-rate logic — the #1 way to leak the label is
    computing rates over the whole dataset instead of as-of, and that must be
    consistent regardless of which model consumes it."""
    missing = [c for c in HEADER_REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")

    # Carry any EXPLICIT label column through the projection. Without this the
    # LABEL_COLUMNS branch in _derive_label is unreachable — the column is dropped
    # here, before that function ever sees the header — and the error message it
    # raises ("add a 0/1 column named one of [...]") advises a fix that cannot work.
    # That matters most for exactly the export that needs it: an unrecognised status
    # vocabulary, where the explicit column is the only escape hatch.
    label_cols = [c for c in LABEL_COLUMNS if c in raw.columns]
    header = _fill_optional(raw, HEADER_OPTIONAL)[HEADER_COLS + label_cols] \
        .drop_duplicates(subset="quotationID").copy()
    header["quoteDate"] = pd.to_datetime(header["quoteDate"], errors="coerce")

    # collapse revisions: keep only the FINAL revision per quote family. With no
    # parentQuotationID every quote roots to itself and this is a no-op, which is
    # the correct reading of an export that has no revision workflow.
    header["_root"] = header["quotationID"].map(_resolve_family_root(header))
    final_ids = (header.sort_values("revisionNumber")
                       .groupby("_root")["quotationID"].last())
    header = header[header["quotationID"].isin(set(final_ids))].reset_index(drop=True)

    header["won"], _label_source = _derive_label(header)
    # Aging only ever applies to quotes the label step left genuinely UNDECIDED.
    # If the label came from an explicit column, an unlabelled row means "no
    # outcome recorded", not "sent and ignored", so ageing it out would invent
    # losses wholesale — which is exactly what happened to a WIN/LOSS export.
    if _label_source == "statusName":
        header = _age_out_open_quotes(header, lost_after_days)
    header["is_closed"] = header["won"].notna()
    header = header[header["is_closed"]].reset_index(drop=True)  # drop what is still live

    header = header.sort_values("quoteDate").reset_index(drop=True)
    header["contact_win_rate"] = _asof_rate(header, "customerID")
    header["salesrep_win_rate"] = _asof_rate(header, "salesRepID")
    return header


def build(raw: pd.DataFrame, tenant: str, *,
          lost_after_days: int | None = OPEN_QUOTE_LOST_AFTER_DAYS) -> tuple[pd.DataFrame, dict]:
    header = _build_header(raw, lost_after_days)
    li = _fill_optional(raw, LINE_OPTIONAL)
    li = li[li["quotationID"].isin(header["quotationID"])][LINE_COLS].copy()
    li["effective_price"] = _effective_price(li)
    li["ratio"] = _price_ratio(li)
    # lineTotal is a cache; when absent (or null) recompute it rather than
    # weighting the whole quote at zero.
    li["amt"] = pd.to_numeric(li["lineTotal"], errors="coerce").fillna(
        pd.to_numeric(li["quantity"], errors="coerce") * li["effective_price"]).fillna(0.0)

    grp = li.groupby("quotationID")
    agg = pd.DataFrame({
        "total_quantity": grp["quantity"].apply(lambda s: float(pd.to_numeric(s, errors="coerce").sum())),
        "line_count": grp.size(),
        "n_products": grp["productID"].nunique(),
        "mean_price_ratio": grp["ratio"].mean(),
        "min_price_ratio": grp["ratio"].min(),
        "wtd_price_ratio": grp.apply(lambda d: float(np.average(d["ratio"].fillna(1.0), weights=d["amt"]))
                                     if d["amt"].sum() > 0 else float(d["ratio"].mean()), include_groups=False),
    })

    out = header.merge(agg, left_on="quotationID", right_index=True, how="left")
    out["avg_discount_pct"] = (1.0 - out["mean_price_ratio"].fillna(1.0)) * 100.0
    out["grand_total"] = pd.to_numeric(out["grandTotal"], errors="coerce")
    # contact_win_rate / salesrep_win_rate already computed as-of in _build_header,
    # and preserved through this left-merge (header keeps its quoteDate-sorted order)
    rate_lookup = {
        "contact_win_rate": _entity_rate_lookup(out, "customerID"),
        "salesrep_win_rate": _entity_rate_lookup(out, "salesRepID"),
    }
    out["tenant"] = tenant
    out = out.rename(columns={"leadTimeDays": "lead_time_days"})

    cols = ["quotationID", "tenant", "grand_total", "total_quantity", "line_count",
            "region", "industry", "lead_time_days", "quoteDate", "won",
            "wtd_price_ratio", "mean_price_ratio", "min_price_ratio", "avg_discount_pct",
            "n_products", "contact_win_rate", "salesrep_win_rate"]
    return out[cols], rate_lookup


def build_line_frame(raw: pd.DataFrame, tenant: str, *,
                     lost_after_days: int | None = OPEN_QUOTE_LOST_AFTER_DAYS) -> pd.DataFrame:
    """gold_line_win.csv-shaped frame: one row per LINE ITEM of every DECIDED quote
    (won AND lost, unlike build_price_frame which is won-only) — feeds the
    m1_quote_line_win classifier (per-product win probability).

    The ground truth (won/lost) is only recorded per QUOTATION, not per line, so
    there is no genuine per-line outcome to train on. This BROADCASTS the quote's
    label onto every one of its lines: every line in a won quote counts as a
    'won' example for that line, every line in a lost quote as 'lost'. That is a
    deliberate modelling choice (confirmed), not an oversight — it lets the model
    learn how a line's OWN price_ratio/product/region relate to the quote's
    outcome, even though it can't distinguish which specific line 'caused' a loss.

    Reuses _build_header()'s as-of contact_win_rate/salesrep_win_rate (same
    historical rates the WIN model uses)."""
    header = _build_header(raw, lost_after_days)
    # quoteDate is carried through so the trainer can hold out FORWARD IN TIME —
    # production retrains monthly and predicts forward, and a random holdout
    # flatters that. quote_total / product_type / payment_terms are the remaining
    # BRD "Training Data Elements" this export can supply; the trainer picks up
    # whichever are non-null and logs the ones that are not.
    # customer recency / prior-quote count, computed on the HEADER so a multi-line
    # quote counts once. Both are strictly prior: cumcount and diff over a
    # date-sorted frame never see the current quote.
    hh = header.sort_values("quoteDate").copy()
    hh["customer_prior_quotes"] = hh.groupby(hh["customerID"].astype(str)).cumcount().astype(float)
    hh["customer_recency_days"] = (hh.groupby(hh["customerID"].astype(str))["quoteDate"]
                                     .diff().dt.days.fillna(-1.0))
    header = hh.sort_index()

    hcols = ["quotationID", "customerID", "region", "industry", "leadTimeDays", "quoteDate",
             "contact_win_rate", "salesrep_win_rate", "won", "paymentTerms",
             "customer_prior_quotes", "customer_recency_days"]
    h = header[hcols].rename(columns={"paymentTerms": "payment_terms"})

    li = _fill_optional(raw, LINE_OPTIONAL)
    keep = ["quotationID", "productID", "quantity", "unitPrice", "salesPrice",
            "negotiatedSalesPrice", "materialSpec"] + (["listPrice"] if "listPrice" in li.columns else [])
    li = li[li["quotationID"].isin(header["quotationID"])][keep].copy()
    li["price_ratio"] = _price_ratio(li)
    li["list_price"] = _price_basis(li)[0]
    # materialSpec IS the BRD's "Product Type (Goods / Services / Combo)" at a
    # finer grain — the only product-taxonomy column these exports carry, and the
    # cold-start fallback for a product with too few comparable quotations.
    li = li.rename(columns={"materialSpec": "product_type"})

    # quote_total is SUMMED FROM THE LINES, not taken from the header's grandTotal.
    # Both carry the same information (corr 1.0 on the all-verticals export, since
    # tax is a flat multiplier), but only this one is computable at serving time
    # from what a rep has already typed into the Products table — grandTotal is not
    # known until tax is applied, and asking the user to type it by hand is how you
    # get a required field nobody can fill. live_features uses the identical
    # formula, so train and serve cannot drift.
    li["_amt"] = (pd.to_numeric(li["quantity"], errors="coerce") * _effective_price(li))
    totals = li.groupby("quotationID")["_amt"].sum()
    li = li.drop(columns=["_amt"])

    out = li.merge(h, on="quotationID", how="left")
    out["quote_total"] = out["quotationID"].map(totals)
    out["tenant"] = tenant
    out = out.dropna(subset=["price_ratio", "won"]).reset_index(drop=True)

    # --- history features, all strictly prior to this quote's date -----------
    # Absolute margin over cost ranks mid-pack in SHAP because it is not
    # comparable across a catalogue whose unit costs span orders of magnitude.
    # What a buyer reacts to is RELATIVE price: is this dearer than we usually
    # sell this product for, or than this customer usually pays?
    out["price_vs_product"] = _prior_ratio(out, "productID", "price_ratio")
    out["price_vs_customer"] = _prior_ratio(out, "customerID", "price_ratio")
    out["product_win_rate"] = _prior_win_rate(out, "productID")
    out["value_vs_customer"] = _prior_ratio(out, "customerID", "quote_total")
    out["leadtime_vs_product"] = _prior_ratio(out, "productID", "leadTimeDays")
    # deal shape — no history needed, known the moment the quote is built
    out["line_share"] = ((pd.to_numeric(out["quantity"], errors="coerce")
                          * _effective_price(out)) / out["quote_total"]).clip(0, 1).fillna(0.0)
    out["quote_month"] = out["quoteDate"].dt.month.astype(float)

    cols = ["tenant", "quotationID", "quoteDate", "productID", "quantity", "unitPrice",
            "list_price", "price_ratio", "leadTimeDays", "region", "industry", "product_type",
            "payment_terms", "quote_total", "contact_win_rate", "salesrep_win_rate",
            "price_vs_product", "price_vs_customer", "product_win_rate",
            "value_vs_customer", "leadtime_vs_product", "line_share", "quote_month",
            "customer_prior_quotes", "customer_recency_days", "won"]
    out = out[cols]
    # drop optional columns that this export could not supply at all, so the
    # trainer's "not present" log stays truthful instead of listing all-null ones
    return out.drop(columns=[c for c in ("product_type", "payment_terms", "quote_total")
                             if out[c].isna().all()])


def build_price_frame(raw: pd.DataFrame, tenant: str, *,
                      lost_after_days: int | None = OPEN_QUOTE_LOST_AFTER_DAYS) -> pd.DataFrame:
    """gold_price_band.csv-shaped frame: one row per LINE ITEM of a WON quote only —
    the price band answers "what price ratio actually closed a deal," so lost/open
    quotes' pricing is excluded (same principle as db_features.py::build_price_frame).

    Reuses _build_header()'s as-of contact_win_rate — the same historical customer
    rate the WIN model uses, joined onto that customer's won lines.

    materialSpec is intentionally NOT derived here even though PRICE_CATEG lists it
    (csv_price.py) — it's sourced from product master data via the DB path instead;
    the adaptive trainer (_select_features) simply skips a feature that isn't present,
    so training still works, just without that one feature from this source."""
    header = _build_header(raw, lost_after_days)
    won = header.loc[header["won"] == 1.0,
                      ["quotationID", "region", "leadTimeDays", "quoteDate", "contact_win_rate"]]

    li = _fill_optional(raw, LINE_OPTIONAL)
    li = li[li["quotationID"].isin(won["quotationID"])][
        ["quotationID", "quotationLineItemID", "productID", "productName",
         "quantity", "unitPrice", "salesPrice", "negotiatedSalesPrice"]
    ].copy()
    li["price_ratio"] = _price_ratio(li)

    out = li.merge(won, on="quotationID", how="left")
    out["tenant"] = tenant
    out = out.dropna(subset=["price_ratio"]).reset_index(drop=True)

    cols = ["tenant", "quotationID", "quotationLineItemID", "quoteDate", "productID",
            "productName", "region", "quantity", "unitPrice", "leadTimeDays",
            "contact_win_rate", "price_ratio"]
    return out[cols]


def build_mil_frame(raw: pd.DataFrame, tenant: str, *,
                    lost_after_days: int | None = OPEN_QUOTE_LOST_AFTER_DAYS) -> pd.DataFrame:
    """One row per LINE of every CLOSED quote (won AND lost) for the Noisy-OR
    Multiple Instance Learning model (m1_quote/csv_mil.py). Each quotationID
    is a BAG of its lines; `won` is broadcast per bag and used ONLY as the
    bag-level label at training time (never treated as a per-line label,
    unlike build_line_frame) — the MIL model combines per-line predictions
    into the bag prediction itself via Noisy-OR, so it needs the raw lines
    grouped by quotationID, not an aggregated or per-line-labelled frame.

    Keeps the notebook's original raw feature set (salesPrice, materialSpec,
    discountPercent) rather than the price_ratio/contact_win_rate derived for
    the other models — csv_mil.py engineers its own margin_ratio/log_quantity
    and one-hot-encodes categoricals directly from these raw columns."""
    header = _build_header(raw, lost_after_days)
    hcols = ["quotationID", "region", "industry", "leadTimeDays", "salesRepID", "won"]
    h = header[hcols]

    li_cols = ["quotationID", "productID", "productName", "quantity", "unitPrice",
               "salesPrice", "negotiatedSalesPrice", "materialSpec", "discountPercent"]
    li = _fill_optional(raw, LINE_OPTIONAL)
    li = li[li["quotationID"].isin(header["quotationID"])][li_cols].copy()
    li["salesPrice"] = _effective_price(li)
    li = li.drop(columns=["negotiatedSalesPrice"])

    out = li.merge(h, on="quotationID", how="left")
    out["tenant"] = tenant
    out = out.dropna(subset=["salesPrice", "unitPrice", "won"]).reset_index(drop=True)

    cols = ["tenant", "quotationID", "productID", "productName", "quantity", "unitPrice",
            "salesPrice", "materialSpec", "discountPercent", "leadTimeDays", "region",
            "industry", "salesRepID", "won"]
    return out[cols]
