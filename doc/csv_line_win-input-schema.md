# `csv_line_win` input schema

Two schemas, and they are not the same thing.

1. The **raw export** you upload — the CSV. `raw_ingest.build_line_frame()` reads it.
2. The **training frame** `csv_line_win.train()` actually consumes — the output of
   that builder. You only construct this yourself if you bypass the CSV path.

Verified against `dataset/row/price_driven_quotes.csv`: 27,302 raw rows in,
26,843 frame rows out, 26 columns.

---

## 1. Raw export (what you upload)

### Required — the upload gate rejects the file without these

| Column | Type | Notes |
|---|---|---|
| `quotationID` | str | the grouping key; all lines of a quote share it |
| `productID` | str | categorical feature and the "comparable" unit |
| `quantity` | int | |
| `unitPrice` | float | **unit COST, not the price charged** |
| `salesPrice` | float | the price quoted to the customer |
| `industry` | str | |
| `customerID` | str | drives `contact_win_rate` and the as-of customer features |
| `region` | str | |
| `salesRepID` | str | drives `salesrep_win_rate` |
| `quoteDate` | date | makes the holdout forward in time; without it the split is random-grouped |
| `leadTimeDays` | int | |
| `statusName` | str | the outcome — see label derivation below |
| `grandTotal` | float | quote total |

### Optional — absent is filled with a neutral value, present trains a better model

| Column | Fallback when absent |
|---|---|
| `listPrice` | **not declared in `RAW_OPTIONAL_COLS` — see the gap below.** Falls back to `unitPrice` as the price basis |
| `materialSpec` | carried through as `product_type`; absent → unknown level |
| `paymentTerms` | categorical feature when present |
| `negotiatedSalesPrice` | falls back to `salesPrice` |
| `lineTotal` | quantity x effective price |
| `parentQuotationID` | no revision chains — every quote is its own family |
| `revisionNumber` | assumed 1 |
| `customerName` | display only, never a feature |
| `productName` | display only |
| `discountPercent` | MIL model only |
| `quotationLineItemID` | identifier, price frame only |

### The label

`won` is derived from `statusName` by **word-boundary** match, not substring:

```
win   won win wins confirmed accepted converted order ordered success successful
loss  lost loss lose rejected declined cancelled canceled failed unsuccessful
```

Anything matching neither is left undecided (NaN) and then resolved by the aging
rule. An explicit `target_win` / `won` / `is_won` / `isWon` column, if present,
overrides the derivation. The label is recorded per QUOTE and broadcast onto every
line of that quote — there is no per-line ground truth in this data.

### The one gap worth knowing

`listPrice` is read by `_price_basis()` and is the single most consequential column
in the feature set — it decides whether `price_ratio` is measured against list
price or against cost — but it is **not** in `LINE_OPTIONAL`, so it never appears
in the upload preview's optional-columns list. An export that has it is used
correctly; an export that omits it silently trains on a cost basis and nothing in
the UI says so.

---

## 2. Training frame (what `train()` consumes)

26 columns. Of these, **21 are booster features**; the other five are the label,
the split keys and the band's price basis.

### Features — 21

| Group | Column | dtype |
|---|---|---|
| Line numeric | `quantity` | int64 |
| | `unitPrice` | float64 |
| | `price_ratio` | float64 |
| | `leadTimeDays` | int64 |
| | `contact_win_rate` | float64 |
| | `salesrep_win_rate` | float64 |
| Line categorical | `productID` | object |
| | `region` | object |
| | `industry` | object |
| BRD numeric | `quote_total` | float64 |
| BRD categorical | `product_type` | object |
| | `payment_terms` | object |
| As-of history | `price_vs_product` | float64 |
| | `price_vs_customer` | float64 |
| | `product_win_rate` | float64 |
| | `value_vs_customer` | float64 |
| | `leadtime_vs_product` | float64 |
| | `line_share` | float64 |
| | `quote_month` | float64 |
| | `customer_prior_quotes` | float64 |
| | `customer_recency_days` | float64 |

Every history feature is computed strictly **before** its own quote's date, so
none can see its own outcome.

### Not features — 5

| Column | dtype | Role |
|---|---|---|
| `won` | float64 | the label, 0.0 / 1.0 |
| `quotationID` | object | the independence unit — the fit/calibrate/score split groups on it |
| `quoteDate` | datetime64[ns] | orders the temporal split |
| `list_price` | float64 | what the recommended ratio is multiplied by to get money. Declared in the MLflow signature so serving carries it — MLflow drops unnamed columns before `predict()` |
| `tenant` | object | metadata |

### Declared but absent from this export

`days_to_expiry` (`BRD_NUMERIC`, from the BRD's Expiration Date) and `below_cost`
(`DERIVED_NUMERIC`). `_select_features()` takes the intersection with the frame's
columns, so a missing declared feature is skipped, not an error. `below_cost` is
additionally gated on `_below_cost_helps()` — it is dropped when the data shows no
distress dip.

### How the price basis is decided

```python
price_basis = "listPrice" if not np.allclose(df["list_price"], df["unitPrice"]) \
              else "unitPrice"
```

So the basis is inferred from whether the two columns actually differ, not from
whether `listPrice` was supplied. Supply a `listPrice` equal to `unitPrice` on
every row and the model records a cost basis.

---

## Minimum viable export

13 required columns, plus `listPrice` (not required, but the model is materially
worse without it) and `materialSpec` and `paymentTerms` if you have them:

```
quotationID,lineNumber,productID,quantity,unitPrice,salesPrice,listPrice,
materialSpec,industry,customerID,region,salesRepID,quoteDate,leadTimeDays,
paymentTerms,statusName,grandTotal
```

`lineNumber` is not read by `build_line_frame` — it is in the export for human
legibility only.
