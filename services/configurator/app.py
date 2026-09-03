"""ML Model Configurator API (FastAPI) — backs the Configurator UI.

Endpoints (M1 Smart Quote Optimiser: three model cards — quote-level Classification,
Regression price band, and per-product line-level Classification):
  GET  /api/{tenant}/models                         list cards + status/metrics
  GET  /api/{tenant}/models/{key}/predict-schema    fields for the Test-Predictions sidebar
  POST /api/{tenant}/models/{key}/predict           run a what-if prediction
  GET  /api/{tenant}/models/{key}/predict-schema-raw  raw (client-facing) test-input columns
  POST /api/{tenant}/models/{key}/predict-raw       predict from raw columns, not ratios
  POST /api/{tenant}/models/{key}/retrain/preview   upload CSV -> preview + column validation
  GET  /api/{tenant}/models/{key}/retrain/db-columns connect DB -> columns we train from
  POST /api/{tenant}/models/{key}/train             start training (bg) -> run_id
  GET  /api/train/{run_id}                           live logs + result
  POST /api/{tenant}/models/{key}/publish           champion gate (only if it beats current)
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mlflow.exceptions import MlflowException

from maxxflow_mlops.naming import GLOBAL_TENANT, registered_model_name
from maxxflow_mlops.registry import MLflowRegistry
from m1_quote import csv_line_win, csv_mil, csv_price, csv_win
from m1_quote.live_features import (
    line_win_features_from_raw,
    mil_features_from_raw,
    price_features_from_raw,
    quote_features_from_raw,
)
from m1_quote.raw_ingest import (
    MIL_OPTIONAL_COLS,
    OPTION_GRAPH_PATH,
    MIL_REQUIRED_COLS,
    RATE_LOOKUP_PATH,
    RAW_OPTIONAL_COLS,
    RAW_REQUIRED_COLS,
)
from m2_inventory import csv_training as inventory_csv
from m2_inventory.inventory_dataset import InventoryDatasetBuilder, MODEL_INPUT_COLUMNS
from services.configurator import jobs
from services.configurator.security import AuthContext, install_security
from services.configurator.training_backends import get_training_backend

app = FastAPI(title="MaXXflow ML Model Configurator", version="1.0")
install_security(app)


_default_origins = "http://localhost:3000,http://localhost:3009"
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ALLOWED_ORIGINS", _default_origins).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "x-tenant-slug", "Authorization"],
    max_age=86400,  # cache preflight 24h
)



@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Return the real server error; CORSMiddleware adds an allowed origin only."""
    return JSONResponse(status_code=500,
                        content={"detail": f"{type(exc).__name__}: {exc}"})
_UPLOADS = Path(tempfile.gettempdir()) / "mxf_uploads"
_UPLOADS.mkdir(exist_ok=True)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_INVENTORY_CSV = _REPO_ROOT / "dataset" / "M2data" / "all_verticals_full.csv"
_INVENTORY_ARTIFACTS = _REPO_ROOT / "artifacts" / "m2_inventory"

# --- model catalogue (M1 = three cards) --------------------------------------
def _num(n): return {"name": n, "type": "number"}
def _cat(n): return {"name": n, "type": "category"}

MODELS = {
    "m1_quote_win": {
        "title": "Smart Quote Optimiser — Classification", "model_type": "Classification Model",
        "description": "Predicts a calibrated Win Probability (0–100%) for a quotation from deal "
                       "shape, pricing ratios and buyer history.",
        "predicts": "won (win probability)", "trainer": csv_win,

        "required": RAW_REQUIRED_COLS,
        "optional": RAW_OPTIONAL_COLS,
        "raw_upload": True,
        "fields": [_num(c) for c in ["grand_total", "total_quantity", "line_count", "n_products",
                   "wtd_price_ratio", "mean_price_ratio", "min_price_ratio", "avg_discount_pct",
                   "contact_win_rate", "salesrep_win_rate"]] + [_cat("region"), _cat("industry")],
        
        "header_fields": [_cat("customerID"), _cat("salesRepID"), _cat("region"), _cat("industry"),
                           _num("leadTimeDays")],
        "line_fields": [_cat("productID"), _num("quantity"), _num("unitPrice"),
                         _num("salesPrice"), _num("negotiatedSalesPrice")],
        "primary_metric": "accuracy",
    },
    "m1_quote_price": {
        "title": "Smart Quote Optimiser — Regression", "model_type": "Regression Model",
        "description": "Recommends a price band (P25/P50/P75 of price_ratio) per line, benchmarked "
                       "against the empirical baseline and clamped to ±30% of list price.",
        "predicts": "price_ratio (price band)", "trainer": csv_price,
        
        "required": RAW_REQUIRED_COLS,
        "optional": RAW_OPTIONAL_COLS,
        "raw_upload": True,
        "fields": [_num(c) for c in ["quantity", "unitPrice", "leadTimeDays", "contact_win_rate"]]
                  + [_cat("productID"), _cat("materialSpec"), _cat("region")],
        
        "header_fields": [_cat("customerID"), _cat("region"), _num("leadTimeDays")],
        "line_fields": [_cat("productID"), _num("quantity"), _num("unitPrice")],
        "primary_metric": "coverage",
    },
    "m1_quote_line_win": {
        "title": "Smart Quote Optimiser — Per-Product Win & Price", "model_type": "Classification Model",
        "description": "Predicts a calibrated Win Probability, recommended Price Band and a "
                       "Confidence Level for EACH product line in a quote — not just the quote "
                       "overall (unlike m1_quote_win). Ground truth is only recorded per quotation, "
                       "so a quote's outcome is broadcast onto every one of its lines for training.",
        "predicts": "won (per-product win probability, bundled with the price-model's band)",
        "trainer": csv_line_win,
        "required": RAW_REQUIRED_COLS,
        "optional": RAW_OPTIONAL_COLS,
        "raw_upload": True,
        # quote_total / product_type / payment_terms are the remaining BRD
        # "Training Data Elements" build_line_frame now carries through when the
        # export has them (grandTotal / materialSpec / paymentTerms). The trainer
        # skips any that are absent, so this list is a superset by design.
        # list_price is not a booster feature — it is what the price band multiplies
        # the recommended RATIO by. It must be declared so _coerce types it and the
        # logged signature carries it; MLflow drops unnamed columns before predict.
        "fields": [_num(c) for c in ["quantity", "unitPrice", "price_ratio", "leadTimeDays",
                   "contact_win_rate", "salesrep_win_rate", "quote_total", "list_price"]]
                  + [_cat("productID"), _cat("region"), _cat("industry"),
                     _cat("product_type"), _cat("payment_terms")],
        # raw, client-facing test-input columns — same shape as m1_quote_win's (the
        # model needs the PROPOSED sale price to derive price_ratio), one prediction
        # PER product line like m1_quote_price. negotiatedSalesPrice stays on the
        # form but is optional: blank falls back to salesPrice, matching ingest.
        # NB: no grandTotal here. quote_total is SUMMED FROM THE LINES by
        # live_features (same formula raw_ingest uses at training time), so the
        # user never types a quote total that the Products table already implies.
        "header_fields": [_cat("customerID"), _cat("salesRepID"), _cat("region"), _cat("industry"),
                           _num("leadTimeDays"), _cat("paymentTerms")],
        # listPrice matters: the model measures price against LIST when the training
        # export had one. A call without it is scored on a different scale than the
        # model learned, which reads as a confident answer and is not one.
        # negotiatedSalesPrice and materialSpec are deliberately NOT collected on this
        # card. MaXXFlow records a single client-facing price rather than a separate
        # negotiated figure, and materialSpec is product master data a rep should not
        # be retyping. live_features already treats both as optional, so omitting them
        # breaks nothing — product_type just falls back to the unknown level on a
        # champion that happened to be trained with it.
        "line_fields": [_cat("productID"), _num("quantity"), _num("unitPrice"),
                         _num("salesPrice"), _num("listPrice")],
        "primary_metric": "accuracy",
    },
    "m1_quote_mil": {
        "title": "Smart Quote Optimiser — MIL (Noisy-OR)", "model_type": "Classification Model",
        "description": "Multiple Instance Learning: treats each quote as a bag of line-item "
                       "instances and combines their latent win probabilities via Noisy-OR "
                       "(product), trained end-to-end from quote-level outcomes only — never a "
                       "per-line label. Reports an MC-Dropout confidence score and a "
                       "Deal-Breaker flag per line; the recommended price band is bundled in "
                       "from the separately-trained m1_quote_price champion (this network's own "
                       "price sensitivity is not yet validated — see csv_mil.py).",
        "predicts": "won (per-line + quote-level win probability, bundled with the "
                    "price-model's band)",
        "trainer": csv_mil,
        # Same raw combined export as the other cards; jobs.py runs it through
        # m1_quote.raw_ingest.build_mil_frame() (won + lost lines, grouped into
        # quote-level bags — needs materialSpec/discountPercent too, hence the
        # wider MIL_REQUIRED_COLS rather than the other cards' RAW_REQUIRED_COLS).
        "required": MIL_REQUIRED_COLS,
        "optional": MIL_OPTIONAL_COLS,
        "raw_upload": True,
        "fields": [_num(c) for c in ["quantity", "unitPrice", "salesPrice", "discountPercent", "leadTimeDays"]]
                  + [_cat("productID"), _cat("materialSpec"), _cat("region"), _cat("industry"), _cat("salesRepID")],
        # raw, client-facing test-input columns — all submitted lines are scored
        # together as ONE quote (one Noisy-OR bag), unlike the per-line-independent
        # m1_quote_price/m1_quote_line_win cards.
        "header_fields": [_cat("salesRepID"), _cat("region"), _cat("industry"), _num("leadTimeDays")],
        "line_fields": [_cat("productID"), _num("quantity"), _num("unitPrice"), _num("salesPrice"),
                         _cat("materialSpec"), _num("discountPercent")],
        "primary_metric": "accuracy",
    },
    "m2_inventory": {
        "title": "Predictive Inventory Alerts", "model_type": "Hazard Classification",
        "description": "Compares Logistic Regression, Random Forest, LightGBM and XGBoost "
                       "to predict product stockout risk over the next 30 and 60 days.",
        "predicts": "30-day and 60-day stockout risk", "trainer": inventory_csv,
        "required": sorted(InventoryDatasetBuilder.required_columns),
        "optional": [
            column for column in MODEL_INPUT_COLUMNS
            if column not in InventoryDatasetBuilder.required_columns
        ],
        "raw_upload": False,
        "fields": [],
        "primary_metric": "weekly_auc",
    },
}


def _model_or_404(key: str) -> dict:
    if key not in MODELS:
        raise HTTPException(404, f"unknown model {key}")
    return MODELS[key]


def _data_tenant(tenant: str) -> str:
    """Whose SCHEMA to read, as opposed to whose MODEL this is.

    The UI's tenant is a model-OWNER slug, and its default is ``global`` — the
    shared base model every new tenant inherits (naming.GLOBAL_TENANT). That is
    a registry concept. There is no ``tenant_global`` schema in the Prisma design
    and there should not be one: rows belong to a real tenant.

    The distinction never came up on the CSV path, where the data arrives in the
    upload and the tenant is only a label. The DB path needs both, and conflating
    them produced "No schema for tenant global — resolved to public", which reads
    as a provisioning failure and is really a category error.

    So a global-owned model reads DEFAULT_TENANT's schema. Its registered name
    stays ``t_global__m_...`` — trained on demo data, owned by everyone, which is
    what a shared base model is. A tenant-owned model reads its own schema.
    """
    from maxxflow_core.settings import get_settings
    return get_settings().default_tenant if tenant == GLOBAL_TENANT else tenant


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/{tenant}/models")
def list_models(tenant: str):
    reg = MLflowRegistry()
    cards = []
    for key, m in MODELS.items():
        own = registered_model_name(tenant, key)
        resolved = reg.resolve_champion_name(tenant=tenant, module=key)
        serving_name, is_base = resolved if resolved else (own, False)
        champ = m["trainer"]._champion_metrics(reg, serving_name) if resolved else None
        status = ("Base model (shared)" if is_base
                  else "Published" if champ else "Not trained")
        if key == "m2_inventory" and not champ and (
            _INVENTORY_ARTIFACTS / "training_summary.json"
        ).exists():
            status = "CSV model ready"
        cards.append({
            "key": key, "title": m["title"], "model_type": m["model_type"],
            "description": m["description"], "predicts": m["predicts"],
            "status": status, "champion": champ,
            "registered_name": own,          # where THIS tenant's training registers
            "serving_name": serving_name,     # what currently answers predictions
            "base_model": is_base,            # True = served by the shared global base
        })
    return {"tenant": tenant, "models": cards}


def _champion_categories(tenant: str, key: str) -> dict:
    """The trained model's known category vocab (e.g. region/industry) — used to
    populate dropdown options. Only features the MODEL itself was trained on have
    a vocab; raw identifiers like customerID/salesRepID/productID never do, and are
    left as free-text fields on purpose (the client can type any real ID)."""
    try:
        reg = MLflowRegistry()
        resolved = reg.resolve_champion_name(tenant=tenant, module=key)
        model = reg.load_champion(name=resolved[0])
        return getattr(model.unwrap_python_model(), "categories", {}) or {}
    except Exception:
        return {}


# Raw, client-facing form column -> the feature name the model learned it under.
# Without this a free-text box lets a user submit "15" for a model whose vocab is
# {"15 Days", "21 Days", "Immediate"}, which scores as an unknown category and
# silently drops the feature.
_FIELD_TO_FEATURE = {"paymentTerms": "payment_terms", "materialSpec": "product_type"}


def _load_option_graph() -> dict:
    """industry -> productID -> materialSpec, written by the last retrain. Missing
    or unreadable means "no cascade", never an error: the dropdowns simply show
    every value, exactly as they did before."""
    try:
        return json.loads(OPTION_GRAPH_PATH.read_text())
    except Exception:
        return {}


def _apply_option_graph(fields: list[dict], graph: dict) -> list[dict]:
    """Attach the parent field and the per-parent option lists so the client can
    narrow a dropdown as its parent changes. `options` is left in place as the
    unfiltered fallback for a parent value the graph has never seen."""
    out = []
    for f in fields:
        f = dict(f)
        spec = graph.get(f["name"])
        if spec and f["type"] == "category":
            f["depends_on"] = spec["parent"]
            f["options_by"] = spec["options_by"]
            if not f.get("options"):
                # no trained vocab for this column (e.g. materialSpec is not a
                # feature of every card) — union the graph so the box is still a
                # dropdown rather than free text
                f["options"] = sorted({v for vs in spec["options_by"].values() for v in vs})[:500]
        out.append(f)
    return out


def _enrich_options(fields: list[dict], cats: dict) -> list[dict]:
    out = [dict(f) for f in fields]
    for f in out:
        vocab = cats.get(f["name"]) or cats.get(_FIELD_TO_FEATURE.get(f["name"], ""))
        if f["type"] == "category" and vocab is not None:
            f["options"] = [str(c) for c in vocab][:200]
    return out


@app.get("/api/{tenant}/models/{key}/predict-schema")
def predict_schema(tenant: str, key: str):
    m = _model_or_404(key)
    fields = _enrich_options(m["fields"], _champion_categories(tenant, key))
    return {"model": key, "fields": fields}


def _none_if_nan(v):
    """NaN is not valid JSON and `float('nan')` serialises to a bare `NaN` token
    that strict parsers reject. It is also the model's way of saying "withheld" —
    the price band could not be computed, or the score must not be shown. None
    carries that meaning across the wire; NaN carries a parse error."""
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
    except TypeError:
        pass
    return v


def _coerce(df: pd.DataFrame, fields: list[dict]) -> pd.DataFrame:
    """Coerce columns to the served signature dtypes: category -> str, else numeric."""
    for f in fields:
        if f["name"] in df.columns:
            df[f["name"]] = (df[f["name"]].astype(str) if f["type"] == "category"
                             else pd.to_numeric(df[f["name"]], errors="coerce").astype(float))
    return df


def _load_champion(reg: MLflowRegistry, tenant: str, key: str):
    """Resolve + load a model's champion (own, else global base). Returns None
    (not an exception) when there's no champion yet — callers decide whether
    that's fatal (a direct /predict call) or gracefully degradable (bundling
    another model's output alongside one that IS trained)."""
    resolved = reg.resolve_champion_name(tenant=tenant, module=key)
    if resolved is None:
        return None
    serving_name, is_base = resolved
    return serving_name, is_base, reg.load_champion(name=serving_name)


@app.post("/api/{tenant}/models/{key}/predict")
def predict(tenant: str, key: str, payload: dict):
    m = _model_or_404(key)
    records = payload.get("records") or [payload]

    reg = MLflowRegistry()
    loaded = _load_champion(reg, tenant, key)
    if loaded is None:
        raise HTTPException(409, "no published (champion) model yet, and no global base model "
                                 "to fall back to — train and publish first")
    serving_name, is_base, model = loaded
    df = _coerce(pd.DataFrame(records), m["fields"])

    preds = model.predict(df)

    return {"predictions": preds.to_dict(orient="records"),
            "serving_name": serving_name, "base_model": is_base}


def _predict_line_win_bundle(tenant: str, header: dict, lines: list[dict], rate_lookup: dict) -> dict:
    """m1_quote_line_win's predict-raw returns ONE bundle per product line: win
    probability, confidence and the recommended price band — all three now from
    the SAME champion.

    The band used to be substituted from the m1_quote_price champion, and the two
    models measure price against different things: line-win against listPrice,
    the price model against unitPrice. On a real request that put the recommended
    price at 30.97 (cost x 1.08) while the win probability was scored at ratio
    0.61 against list — deep in the region where the booster has no splits and
    returns a constant. The panel recommended a price its own probability could
    not score, and reported a confident 89.9% beside it.

    So the line-win model's expected-margin band is now the recommendation, and
    it is swept over that model's informative range, which makes the two halves
    of the panel consistent by construction. The price model's lookup is still
    returned, under `price_model_*`, so the two can be compared in production
    before either is retired. If m1_quote_price has no champion those fields come
    back null with a note rather than failing the request."""
    win_records = line_win_features_from_raw(header, lines, rate_lookup)
    price_records = price_features_from_raw(header, lines, rate_lookup)

    reg = MLflowRegistry()
    win_loaded = _load_champion(reg, tenant, "m1_quote_line_win")
    if win_loaded is None:
        raise HTTPException(409, "no published (champion) model yet for m1_quote_line_win, and no "
                                 "global base model to fall back to — train and publish first")
    win_name, win_is_base, win_model = win_loaded
    win_df = _coerce(pd.DataFrame(win_records), MODELS["m1_quote_line_win"]["fields"])
    try:
        win_preds = win_model.predict(win_df).to_dict(orient="records")
    except MlflowException as e:
        # The champion's logged signature is the serving contract. A mismatch here
        # means it was trained on features this request does not carry — actionable
        # for the caller, so say so rather than 500-ing.
        raise HTTPException(400, f"{win_name} cannot score this input: {e}")

    price_loaded = _load_champion(reg, tenant, "m1_quote_price")
    price_preds = None
    if price_loaded is not None:
        _, _, price_model = price_loaded
        price_df = _coerce(pd.DataFrame(price_records), MODELS["m1_quote_price"]["fields"])
        price_preds = price_model.predict(price_df).to_dict(orient="records")

    predictions = []
    for line, w, p in zip(win_records, win_preds, price_preds or [None] * len(win_records)):
        row = {"productID": line["productID"], "win_probability": w["win_probability"],
               "win_probability_pct": w["win_probability_pct"],
               # statistical CI from the bootstrap ensemble (full-scenario uncertainty) —
               # distinct from `confidence` below, which only reflects the product's own
               # historical row count. Falls back to None on champions trained before
               # this field existed (see LineWinModel.predict's getattr guard).
               "win_probability_ci_low_pct": w.get("win_probability_ci_low_pct"),
               "win_probability_ci_high_pct": w.get("win_probability_ci_high_pct"),
               "n_comparable": w["n_comparable"], "confidence": w["confidence"],
               # a price outside the trained range returns a constant, not an
               # estimate — the caller has to be able to see that
               "price_input_out_of_distribution": w.get("price_input_out_of_distribution"),
               "price_basis": w.get("price_basis"),
               "guardrail_reason": w.get("guardrail_reason"),
               "confidence_message": w.get("confidence_message"),
               # False when the price sits where the model returns a constant.
               # Bind any "trust this number" UI to THIS, not to the percentage.
               "win_probability_reliable": w.get("win_probability_reliable"),
               # NaN -> None: the panel must show the guardrail card instead of a
               # number when the score cannot be trusted.
               "win_probability_display": _none_if_nan(w.get("win_probability_display")),
               # Always a real score, because the recommendation can no longer
               # land outside the range the model can price. This is what the
               # panel should lead with when the rep's own price is unscoreable.
               "win_probability_at_recommended_pct":
                   _none_if_nan(w.get("win_probability_at_recommended_pct")),
               # --- the recommendation, from the SAME model as the probability ---
               "recommended_price_low": _none_if_nan(w.get("recommended_price_low")),
               "recommended_price_mid": _none_if_nan(w.get("recommended_price_mid")),
               "recommended_price_high": _none_if_nan(w.get("recommended_price_high")),
               "price_band_status": w.get("price_band_status"),
               "price_band_message": w.get("price_band_message"),
               # what the band optimises. It is max WIN PROBABILITY subject to
               # never quoting below cost — say so in the payload so the panel can
               # label it and nobody has to infer the objective from the numbers.
               "band_objective": w.get("band_objective"),
               "win_probability_at_low_pct": _none_if_nan(w.get("win_probability_at_low_pct")),
               "win_probability_at_mid_pct": _none_if_nan(w.get("win_probability_at_mid_pct")),
               "win_probability_at_high_pct": _none_if_nan(w.get("win_probability_at_high_pct")),
               # the expected-margin optimum, for comparison only
               "ev_price_low": _none_if_nan(w.get("ev_price_low")),
               "ev_price_mid": _none_if_nan(w.get("ev_price_mid")),
               "ev_price_high": _none_if_nan(w.get("ev_price_high")),
               "band_source": "line_win_max_win_probability"}
        # the price model's lookup, kept alongside for comparison — NOT the
        # recommendation. Note its basis may differ from the win model's.
        row.update({"price_model_low": p["recommended_price_low"],
                    "price_model_mid": p["recommended_price_mid"],
                    "price_model_high": p["recommended_price_high"],
                    "price_model_served_mode": p["served_mode"]}
                   if p is not None else
                   {"price_model_low": None, "price_model_mid": None,
                    "price_model_high": None, "price_model_served_mode": None})
        predictions.append(row)

    note = None if price_preds is not None else ("m1_quote_price has no published model yet — "
                                                  "the comparison fields are omitted. The "
                                                  "recommended band is unaffected: it comes from "
                                                  "the line-win model.")
    return {"predictions": predictions, "serving_name": win_name, "base_model": win_is_base,
            "price_model_note": note}


def _predict_mil_bundle(tenant: str, header: dict, lines: list[dict], rate_lookup: dict) -> dict:
    """m1_quote_mil's own price sweep (recommend_price_band, in csv_mil.py) is not
    served — its win-probability-vs-price curve is close to flat and
    non-monotonic for typical inputs, so the "recommendation" it produces isn't
    trustworthy advisory output (see the price_band_* guardrail flags in its
    prediction). Same bundling pattern as _predict_line_win_bundle: win/quote
    probability + guardrail flags from the m1_quote_mil champion, zipped with
    the recommended price band from the separately-trained, validated
    m1_quote_price champion. If m1_quote_price has no champion yet the price
    fields come back null with a note, same as the line_win bundle."""
    mil_records = mil_features_from_raw(header, lines)
    price_records = price_features_from_raw(header, lines, rate_lookup)

    reg = MLflowRegistry()
    mil_loaded = _load_champion(reg, tenant, "m1_quote_mil")
    if mil_loaded is None:
        raise HTTPException(409, "no published (champion) model yet for m1_quote_mil, and no "
                                 "global base model to fall back to — train and publish first")
    mil_name, mil_is_base, mil_model = mil_loaded
    mil_df = _coerce(pd.DataFrame(mil_records), MODELS["m1_quote_mil"]["fields"])
    mil_preds = mil_model.predict(mil_df).to_dict(orient="records")

    price_loaded = _load_champion(reg, tenant, "m1_quote_price")
    price_preds = None
    if price_loaded is not None:
        _, _, price_model = price_loaded
        price_df = _coerce(pd.DataFrame(price_records), MODELS["m1_quote_price"]["fields"])
        price_preds = price_model.predict(price_df).to_dict(orient="records")

    predictions = []
    for line, m, p in zip(mil_records, mil_preds, price_preds or [None] * len(mil_records)):
        row = {"productID": line["productID"], "win_probability": m["win_probability"],
               "win_probability_pct": m["win_probability_pct"],
               "mc_dropout_confidence": m["mc_dropout_confidence"],
               "quote_win_probability": m["quote_win_probability"],
               "price_band_monotonicity_violated": m["price_band_monotonicity_violated"],
               "price_band_boundary_hit": m["price_band_boundary_hit"],
               "price_band_out_of_distribution": m["price_band_out_of_distribution"],
               "deal_breaker": m["deal_breaker"],
               "n_comparable": m["n_comparable"], "confidence": m["confidence"]}
        row.update({"recommended_price_low": p["recommended_price_low"], "recommended_price_mid": p["recommended_price_mid"],
                    "recommended_price_high": p["recommended_price_high"], "served_mode": p["served_mode"]}
                   if p is not None else
                   {"recommended_price_low": None, "recommended_price_mid": None,
                    "recommended_price_high": None, "served_mode": None})
        predictions.append(row)

    note = None if price_preds is not None else ("m1_quote_price has no published model yet — "
                                                  "price band omitted; train & publish it too.")
    return {"predictions": predictions, "serving_name": mil_name, "base_model": mil_is_base,
            "price_model_note": note}


def _load_rate_lookup() -> dict:
    if not RATE_LOOKUP_PATH.exists():
        # no lookup generated yet — every customer/rep falls back to a neutral 50%
        return {"contact_win_rate": {"_global": 0.5}, "salesrep_win_rate": {"_global": 0.5}}
    return json.loads(RATE_LOOKUP_PATH.read_text())


@app.get("/api/{tenant}/models/{key}/predict-schema-raw")
def predict_schema_raw(tenant: str, key: str):
    """Raw, client-facing test-input columns — what a salesperson actually has on
    hand (customer, region, lead time, product lines), never a computed ratio or
    win-rate."""
    m = _model_or_404(key)
    if "header_fields" not in m:
        raise HTTPException(400, f"{key} has no raw test-input schema yet")
    cats, graph = _champion_categories(tenant, key), _load_option_graph()
    return {"model": key,
            "header_fields": _apply_option_graph(_enrich_options(m["header_fields"], cats), graph),
            "line_fields": _apply_option_graph(_enrich_options(m["line_fields"], cats), graph)}


@app.post("/api/{tenant}/models/{key}/predict-raw")
def predict_raw(tenant: str, key: str, payload: dict):
    """Body: {"header": {...}, "lines": [{...}, ...]}
    Derives the engineered feature row(s) server-side (same math as training), then
    predicts exactly like /predict — the client only ever sees/edits raw columns.
    m1_quote_win aggregates all lines into ONE quote-level prediction; m1_quote_price
    predicts a band PER LINE, so it returns one prediction per submitted line."""
    m = _model_or_404(key)
    if "header_fields" not in m:
        raise HTTPException(400, f"{key} does not support raw-column testing yet")
    header = payload.get("header") or {}
    lines = payload.get("lines") or []
    rate_lookup = _load_rate_lookup()
    try:
        if key == "m1_quote_win":
            records = [quote_features_from_raw(header, lines, rate_lookup)]
        elif key == "m1_quote_line_win":
            return _predict_line_win_bundle(tenant, header, lines, rate_lookup)
        elif key == "m1_quote_mil":
            return _predict_mil_bundle(tenant, header, lines, rate_lookup)
        else:
            records = price_features_from_raw(header, lines, rate_lookup)
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(400, f"invalid input: {e}")
    return predict(tenant, key, {"records": records})


@app.post("/api/{tenant}/models/{key}/retrain/preview")
async def retrain_preview(tenant: str, key: str, file: UploadFile = File(...)):
    m = _model_or_404(key)
    upload_id = uuid.uuid4().hex[:12]
    path = _UPLOADS / f"{upload_id}.csv"
    path.write_bytes(await file.read())
    df = pd.read_csv(path)
    missing = [c for c in m["required"] if c not in df.columns]
    # Optional columns never block an upload: raw_ingest synthesizes a neutral
    # value for each (no revision chain, negotiated price falls back to the quoted
    # one, ...). They are reported so the user can see what their export is
    # leaving on the table rather than discovering it in the training log.
    optional = m.get("optional", [])
    opt_present = [c for c in optional if c in df.columns]
    opt_missing = [c for c in optional if c not in df.columns]
    msg = f"Dataset is missing required columns: {missing}" if missing else (
        "Dataset is valid." if not opt_missing else
        f"Dataset is valid. {len(opt_missing)} optional column(s) absent — these are "
        f"filled in automatically, but supplying them trains a better model: "
        f"{', '.join(opt_missing)}.")
    return {
        "upload_id": upload_id, "filename": file.filename,
        "rows": int(len(df)), "columns": list(df.columns),
        "required_columns": m["required"], "missing_columns": missing,
        "optional_columns": optional, "optional_present": opt_present,
        "optional_missing": opt_missing,
        "valid": not missing,
        "message": msg,
        # to_json (not to_dict) so NaN -> null; raw exports have plenty of NaN
        # (parentQuotationID, negotiatedSalesPrice, ...) that to_dict leaves as
        # float('nan'), which isn't valid JSON and 500s the response.
        "sample": json.loads(df.head(5).to_json(orient="records")),
    }


@app.get("/api/{tenant}/models/{key}/retrain/db-columns")
def db_columns(tenant: str, key: str):
    """Connect option — HONEST report. The gold training columns are DERIVED at
    train time from RAW source tables (quotations, line items, products, master
    data); they are NOT physical columns. So instead of checking each gold feature
    against information_schema (which always reads 'absent'), we report the RAW
    source tables: present? how many rows? — plus the derived features for context.
    Returns connected=false + an error when the replica isn't reachable (no false
    'connected' banner)."""
    _model_or_404(key)
    from m1_quote import db_features
    note = ("The DB path reads these source tables from the tenant read replica and builds the "
            "gold features at training time; the listed features are DERIVED (never physical "
            "columns). Train with POST /train {\"source\":\"db\"}.")
    spec = db_features._SOURCE_SPEC.get(key)
    if spec is None:
        return {"model": key, "connected": False, "error": "no DB source spec for this model",
                "schema": None, "sources": [], "passthrough_columns": [],
                "derived_features": [], "trainable": False, "note": note}
    dt = _data_tenant(tenant)
    owner_note = (f" Model owner is '{tenant}' (the shared base model); rows come from "
                  f"tenant '{dt}'.") if dt != tenant else ""
    try:
        rep = db_features.describe_sources(dt, key)
        rep.update({"model": key, "error": None, "note": note + owner_note,
                    "data_tenant": dt})
        return rep
    except Exception as e:
        return {"model": key, "connected": False,
                "error": f"read replica not reachable ({type(e).__name__}: {e})",
                "schema": None, "sources": [], "data_tenant": dt,
                "passthrough_columns": spec["passthrough"], "derived_features": spec["derived"],
                "trainable": False, "note": note + owner_note}


@app.post("/api/{tenant}/models/{key}/read_data_from_mrp")
def read_data_from_mrp(tenant: str, key: str, payload: dict | None = None):
    """Execute the DB read and hand back the frame the model would train on.

    ``db-columns`` answers "can I connect and do the source tables have rows?".
    This answers the next question — "what do those rows actually look like once
    the features are built?" — because a row count is not a dataset. An export
    can have 1600 quotations and still produce a frame that is empty after the
    label join, or all-one-class, or missing a feature column.

    It calls ``jobs.build_db_frame``, the SAME function ``source="db"`` training
    calls. A preview that re-issued the SELECTs itself would be a second
    implementation of the read path, free to drift from the first, and the whole
    point of showing it is that it is what trains.

    Cost: this builds the full frame, so it is as slow as the feature step of a
    training run (seconds on a demo tenant). It is a POST for that reason — it
    is work, not a lookup — and the response carries ``rows`` for the true size
    with ``sample`` capped at ``limit``.

    Errors come back 200 with ``ok: false``, matching db-columns, so the wizard
    renders a banner instead of a network failure.
    """
    _model_or_404(key)
    payload = payload or {}
    try:
        limit = max(1, min(int(payload.get("limit", 25)), 200))
    except (TypeError, ValueError):
        limit = 25

    if key not in jobs.db_models():
        return {"ok": False, "model": key, "tenant": tenant,
                "error": f"{key} has no DB-path builder — it is CSV-upload only. "
                         f"Models with a DB path: {', '.join(jobs.db_models())}.",
                "rows": 0, "columns": [], "dtypes": {}, "sample": []}

    dt = _data_tenant(tenant)
    t0 = time.perf_counter()
    try:
        df = jobs.build_db_frame(dt, key)
    except Exception as e:
        return {"ok": False, "model": key, "tenant": tenant, "data_tenant": dt,
                "error": f"{type(e).__name__}: {e}",
                "rows": 0, "columns": [], "dtypes": {}, "sample": []}

    rows, cols = int(len(df)), list(df.columns)
    # Empty is not an error — it is the single most useful thing this endpoint
    # can tell you, and it is invisible from a row count on the source tables.
    if rows == 0:
        message = (f"The query ran against tenant_{dt} and returned no rows. "
                   "The source tables have data, so the "
                   "loss is in the join or the label filter — most often quotations whose "
                   "stage/status never resolved to won or lost.")
    else:
        message = (f"{rows} rows x {len(cols)} columns built from schema "
                   f"tenant_{dt}.")

    return {
        "ok": True, "model": key, "tenant": tenant, "data_tenant": dt,
        "rows": rows, "columns": cols,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        # to_json, not to_dict: NaN is not valid JSON and to_dict leaves it as
        # float('nan'), which 500s the response. Same reason as retrain/preview.
        "sample": json.loads(df.head(limit).to_json(orient="records")),
        "sample_size": min(rows, limit),
        "truncated": rows > limit,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "message": message,
    }


@app.post("/api/{tenant}/models/{key}/train")
def train(tenant: str, key: str, payload: dict):
    _model_or_404(key)
    source = payload.get("source", "csv")
    auto_hpo = payload.get("auto_hpo", True)
    if source == "db":                       # Connect-MaXXflow-Database path
        # Goes through the SAME backend factory as the CSV path. This used to be
        # pinned in-process on the grounds that "an AML job would need its own
        # route to that database" — it now has one: the job runs `maxxflow
        # train-db`, reaches Postgres directly, and fetches the password from Key
        # Vault with the compute's managed identity. Pinning it here would have
        # meant TRAIN_BACKEND=azureml silently not applying to half the UI.
        #
        # tenant = who OWNS the model (registry name); data_tenant = whose rows it
        # learns from. For the shared base model those differ; see _data_tenant.
        dt = _data_tenant(tenant)
        run_id = get_training_backend().start(tenant, key, source="db", csv_path=None,
                                              auto_hpo=auto_hpo, data_tenant=dt)
        return {"run_id": run_id, "source": "db", "data_tenant": dt}
    upload_id = payload.get("upload_id")     # Upload-a-File path
    if not upload_id:
        raise HTTPException(400, "provide upload_id (from /retrain/preview) or source='db'")
    path = _UPLOADS / f"{upload_id}.csv"
    if not path.exists():
        raise HTTPException(404, "upload not found — re-run preview")
    run_id = get_training_backend().start(tenant, key, source="csv", csv_path=str(path),
                                          auto_hpo=auto_hpo)
    return {"run_id": run_id, "source": "csv"}


@app.get("/api/train/{run_id}")
def train_status(run_id: str, request: Request):
    """Same {status, logs, result} shape whichever backend ran it, so the UI does
    not need to know where training happened. An in-process run is not in the AML
    backend's map and vice versa, so ask the local dict first and fall through."""
    result = jobs.status(run_id)
    if result.get("status") == "unknown":
        result = get_training_backend().status(run_id)
    auth: AuthContext = request.state.auth
    run_tenant = result.get("tenant")
    if run_tenant not in (None, "?", "global", auth.tenant_slug):
        raise HTTPException(403, "Training run belongs to another tenant")
    return result


@app.post("/api/{tenant}/models/{key}/publish")
def publish(tenant: str, key: str, payload: dict):
    m = _model_or_404(key)
    version = payload.get("version")
    metrics = payload.get("metrics")
    if not version or not metrics:
        raise HTTPException(400, "provide version + metrics from the training result")
    # `force` overrides the champion/challenger COMPARISON gate — an explicit human
    # decision to ship a candidate that is not measurably better. It does not
    # override the data-validation stop, and the forced decision is still recorded
    # in `reasons` so the audit trail says it was overridden and by which check.
    force = bool(payload.get("force", False))
    return m["trainer"].publish(tenant, str(version), metrics, force=force)


@app.get("/api/{tenant}/inventory-dashboard")
def inventory_dashboard(tenant: str):
    """Score the newest CSV snapshot for every product/warehouse combination."""
    builder = InventoryDatasetBuilder()
    try:
        snapshots = builder.load(_INVENTORY_CSV)
        latest = (
            snapshots.sort_values("snapshot_date")
            .groupby(["item_id", "warehouse_id"], as_index=False, sort=False)
            .tail(1)
            .reset_index(drop=True)
        )
        # This dashboard is explicitly CSV-backed.  It uses the selected artifact
        # written by the most recent wizard/CLI CSV run and therefore remains
        # available even when MLflow is offline or the quality gate correctly
        # declined to publish a champion.
        risks = inventory_csv.score_latest_csv_snapshots(
            _INVENTORY_CSV, _INVENTORY_ARTIFACTS
        )
        model_source = "latest CSV evaluation winner"
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(409, f"inventory dashboard is not ready: {exc}") from exc

    risk_columns = [
        "item_id", "warehouse_id", "risk_30d", "risk_60d", "badge_30d",
        "badge_60d", "suppressed", "model_version",
    ]
    scored = latest.merge(risks[risk_columns], on=["item_id", "warehouse_id"], how="left")
    def number(value) -> float:
        return 0.0 if pd.isna(value) else float(value)

    rows = []
    for row in scored.sort_values("risk_30d", ascending=False).itertuples(index=False):
        available = number(row.available_qty)
        reserved = number(row.reserved_qty)
        rows.append({
            "item_id": str(row.item_id),
            "product": str(getattr(row, "item_name", "") or row.item_id),
            "part_number": str(getattr(row, "part_number", "") or ""),
            "warehouse": str(getattr(row, "warehouse_name", "") or row.warehouse_id),
            "quantity": available,
            "reserved_quantity": reserved,
            "available_quantity": available - reserved,
            "reorder_point": number(row.rop),
            "forecast_quantity": number(row.demand_forecast_qty),
            "risk_30d": float(row.risk_30d),
            "risk_60d": float(row.risk_60d),
            "badge_30d": str(row.badge_30d),
            "badge_60d": str(row.badge_60d),
            "suppressed": bool(row.suppressed),
        })
    summary_path = _INVENTORY_ARTIFACTS / "training_summary.json"
    selected_model = None
    model_metrics = []
    selection_warning = None
    if summary_path.exists():
        try:
            training_summary = json.loads(summary_path.read_text())
            selected_model = training_summary["selected_model"]
            selection_warning = training_summary.get("selection_warning")
            metric_names = [
                "algorithm", "calibration_method", "test_rows", "positive_rate",
                "accuracy", "base_rate_accuracy", "accuracy_over_base_rate",
                "weekly_auc", "weekly_brier", "weekly_ece",
                "risk_30d_auc", "risk_30d_brier", "risk_30d_ece",
                "risk_60d_auc", "risk_60d_brier", "risk_60d_ece",
                "quality_floor_passed",
            ]
            model_metrics = [
                {name: candidate.get(name) for name in metric_names}
                | {"selected": candidate.get("algorithm") == selected_model}
                for candidate in training_summary.get("models", [])
            ]
        except (KeyError, OSError, json.JSONDecodeError):
            pass
    return {
        "tenant": tenant,
        "snapshot_date": latest["snapshot_date"].max().date().isoformat(),
        "model_source": model_source,
        "selected_model": selected_model,
        "model_metrics": model_metrics,
        "selection_warning": selection_warning,
        "rows": rows,
        "summary": {
            "products": len(rows),
            "high_risk_30d": sum(row["risk_30d"] >= 0.66 for row in rows),
            "medium_risk_30d": sum(0.33 <= row["risk_30d"] < 0.66 for row in rows),
            "low_risk_30d": sum(row["risk_30d"] < 0.33 for row in rows),
        },
    }


_INVENTORY_FIELD_DESCRIPTIONS = {
    "item_id": "Product/item identifier used to label this what-if prediction.",
    "warehouse_id": "Warehouse identifier; the model learns warehouse-specific behaviour.",
    "snapshot_date": "Date of the inventory snapshot and start of the risk projection.",
    "available_qty": "Physical on-hand quantity before reservations are deducted.",
    "reserved_qty": "Quantity already committed to orders or production.",
    "forecasted_qty": "Forecast quantity supplied by the source inventory export.",
    "rop": "Configured reorder point for this item and warehouse.",
    "demand_forecast_qty": "Expected demand during one weekly hazard interval.",
    "past_due_qty": "Demand already overdue and still requiring inventory.",
    "unit_cost": "Cost of one inventory unit.",
    "unit_of_measurement": "CSV unit code such as EA, M or VIAL.",
    "lead_time_days": "Expected replenishment lead time in days.",
    "computed_reliability": "Primary vendor reliability as a value from 0 to 1.",
    "open_qty": "Quantity currently expected from open purchase orders.",
    "open_po_deadline": "Expected receipt date for the open purchase order; may be blank.",
    "open_po_vendor_reliability": "Open-PO vendor reliability from 0 to 1.",
    "planned_bom_qty": "Planned BOM/manufacturing consumption for the interval.",
    "months_of_history": "Available item history; below 6 months triggers rule-based scoring.",
    "use_rule_based": "Explicitly use the cold-start shortage rule instead of learned hazard.",
    "n_products_using_item": "Number of downstream products that consume this item.",
    "vertical": "Business vertical exactly as represented in the training CSV.",
    "demand_pattern": "Demand regime exactly as represented in the training CSV.",
}
_INVENTORY_PREDICTION_REQUIRED = {
    "item_id", "warehouse_id", "snapshot_date", "available_qty",
    "reserved_qty", "rop", "demand_forecast_qty",
}


def _inventory_field_type(column: str) -> str:
    if column in {"snapshot_date", "open_po_deadline"}:
        return "date"
    if column == "use_rule_based":
        return "boolean"
    if column in {"item_id", "warehouse_id", "unit_of_measurement", "vertical", "demand_pattern"}:
        return "category"
    return "number"


@app.get("/api/{tenant}/models/m2_inventory/predict-schema-csv")
def inventory_predict_schema_csv(tenant: str):
    """Latest CSV product locations and their hidden model-input context."""
    try:
        snapshots = InventoryDatasetBuilder().load(_INVENTORY_CSV)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(409, f"inventory CSV is not ready: {exc}") from exc
    latest = (
        snapshots.sort_values("snapshot_date")
        .groupby(["item_id", "warehouse_id"], as_index=False, sort=False)
        .tail(1)
        .reset_index(drop=True)
    )
    def json_value(column: str, value):
        field_type = _inventory_field_type(column)
        if pd.isna(value):
            return None
        if field_type == "date":
            return pd.Timestamp(value).date().isoformat()
        if field_type == "boolean":
            return bool(value)
        if field_type == "number":
            return float(value)
        return str(value)

    def number(value) -> float:
        return 0.0 if pd.isna(value) else float(value)

    examples = []
    for row in latest.to_dict(orient="records"):
        quantity = number(row.get("available_qty"))
        reserved = number(row.get("reserved_qty"))
        examples.append({
            "product": str(row.get("item_name") or row["item_id"]),
            "part_number": str(row.get("part_number") or ""),
            "warehouse": str(row.get("warehouse_name") or row["warehouse_id"]),
            "item_id": str(row["item_id"]),
            "warehouse_id": str(row["warehouse_id"]),
            "quantity": quantity,
            "reserved": reserved,
            "available": quantity - reserved,
            "reorder_point": number(row.get("rop")),
            "forecast_demand": number(row.get("demand_forecast_qty")),
            "model_input": {
                column: json_value(column, row.get(column))
                for column in MODEL_INPUT_COLUMNS
            },
        })
    examples.sort(key=lambda example: (
        example["product"], example["part_number"], example["warehouse"]
    ))
    return {
        "model": "m2_inventory",
        "tenant": tenant,
        "source": str(_INVENTORY_CSV),
        "examples": examples,
    }


@app.post("/api/{tenant}/models/m2_inventory/predict-csv")
def inventory_predict_csv(tenant: str, payload: dict):
    """Score one what-if record expressed with the exact raw CSV feature names."""
    record = payload.get("record", payload)
    if not isinstance(record, dict):
        raise HTTPException(400, "provide a JSON object in 'record'")
    missing = sorted(
        column for column in _INVENTORY_PREDICTION_REQUIRED
        if record.get(column) in (None, "")
    )
    if missing:
        raise HTTPException(400, f"missing required CSV fields: {missing}")
    try:
        raw = pd.DataFrame([record]).reindex(columns=MODEL_INPUT_COLUMNS)
        if isinstance(raw.loc[0, "use_rule_based"], str):
            raw.loc[0, "use_rule_based"] = raw.loc[0, "use_rule_based"].strip().lower() in {
                "true", "1", "yes",
            }
        model = inventory_csv.load_selected_model(_INVENTORY_ARTIFACTS)
        model_input = inventory_csv._coerce_model_input(raw)
        prediction = model.predict_risk(model_input).iloc[0]
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise HTTPException(409, f"inventory prediction is not ready: {exc}") from exc
    return {
        "tenant": tenant,
        "algorithm": model.algorithm_name,
        "model_source": "latest CSV evaluation winner",
        "prediction": {
            "item_id": str(prediction.get("item_id", record["item_id"])),
            "warehouse_id": str(prediction.get("warehouse_id", record["warehouse_id"])),
            "risk_30d": float(prediction["risk_30d"]),
            "risk_60d": float(prediction["risk_60d"]),
            "badge_30d": str(prediction["badge_30d"]),
            "badge_60d": str(prediction["badge_60d"]),
            "is_rule_based": bool(prediction["is_rule_based"]),
            "suppressed": bool(prediction["suppressed"]),
            "model_version": str(prediction["model_version"]),
        },
    }


# ===========================================================================
# THE UI, SERVED FROM THIS SAME APP
# ===========================================================================
# The Configurator UI is a Vite SPA. It used to be a second container serving
# `vite preview`, with the API's origin baked into the bundle at build time as
# VITE_API_BASE — which meant the frontend image was pinned to one API URL and
# every cross-origin call depended on the permissive CORS above.
#
# Serving the built assets from FastAPI removes both problems: the bundle uses
# RELATIVE paths (`/api/...`), so the same image works behind any hostname, and
# UI and API are same-origin, so CORS stops being load-bearing. It also means
# one image, one Container App, one URL to lock down when the security work
# lands.
#
# Mounted LAST, deliberately. Starlette matches routes in registration order,
# so every /api and /health route above is already claimed before the catch-all
# below sees a request. Adding a new API route after this block would be
# shadowed by it — put new routes above this line.
_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

if _DIST.is_dir():
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    # Hashed asset filenames (Vite writes assets/index-<hash>.js), so these are
    # immutable and safe to cache hard. index.html is NOT cached — it is what
    # points at the current hashes, and a stale one serves a deleted bundle.
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def _spa(full_path: str) -> FileResponse:
        """Serve the SPA, and let client-side routing own unknown paths.

        A real file under dist/ (favicon, manifest) is returned as itself;
        anything else gets index.html so a deep link or a refresh lands in the
        app instead of a 404.

        The /api guard below is NOT redundant with registration order. Order
        stops this handler shadowing routes that EXIST, but an unmatched /api
        path still falls through to here, and answering it with index.html is
        actively harmful: api.ts calls res.json() on every response, so a typo'd
        or removed endpoint would surface as a JSON parse error on a chunk of
        HTML instead of "404 Not Found". Verified before the guard existed —
        GET /api/nope answered 200 text/html."""
        if full_path.startswith(("api/", "docs", "openapi.json", "redoc")):
            raise HTTPException(404, f"no such endpoint: /{full_path}")
        candidate = (_DIST / full_path).resolve()
        if full_path and candidate.is_file() and candidate.is_relative_to(_DIST):
            # is_relative_to guards path traversal: "../../etc/passwd" resolves
            # outside dist/ and falls through to index.html rather than serving.
            return FileResponse(candidate)
        return FileResponse(_DIST / "index.html", headers={"Cache-Control": "no-cache"})
else:
    # No build present — normal when running the API from a checkout for local
    # development, where `npm run dev` serves the UI on :5173 instead. Say so at
    # / rather than 404ing, so "I opened the API and got nothing" is answerable.
    @app.get("/", include_in_schema=False)
    def _no_ui() -> dict:
        return {"api": "up", "ui": f"not built — no {_DIST}. Run `npm run build` in "
                                   f"frontend/, or use `npm run dev` on :5173",
                "docs": "/docs"}
