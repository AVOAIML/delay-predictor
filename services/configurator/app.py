"""ML Model Configurator API (FastAPI) — backs the Configurator UI.

Endpoints (public Configurator cards: Smart Quote Optimiser and Predictive Inventory Alerts):
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

M3 (Production Delay) is not a model card — it trains nothing and has no
champion — so it has its own pair of routes near the bottom of this file:
  POST /api/{tenant}/models/m3_production_delay/batch-review  score+review+cache
  GET  /api/{tenant}/delay-insights                           read the cached result
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mlflow.exceptions import MlflowException

from maxxflow_core.clock import get_clock
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
    price_ratio_for,
)
from m2_inventory import batch_scoring as inventory_batch
from m2_inventory import csv_training as inventory_csv
from m2_inventory.db_prediction import read_db_prediction_snapshots
from m2_inventory.inventory_dataset import InventoryDatasetBuilder, MODEL_INPUT_COLUMNS
from services.configurator import jobs
from services.configurator.result_store import get_training_result_store
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
_COLUMN_PREVIEW_PAGE_SIZE = 8
_SAMPLE_PREVIEW_PAGE_SIZES = (5, 10, 50, 100)
_INVENTORY_DASHBOARD_PAGE_SIZES = (10, 20, 50, 100, 200)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_INVENTORY_CSV = _REPO_ROOT / "dataset" / "M2data" / "all_verticals_full.csv"
_INVENTORY_ARTIFACTS = _REPO_ROOT / "artifacts" / "m2_inventory"

# --- model definitions + public catalogue -----------------------------------
def _num(n): return {"name": n, "type": "number"}
def _cat(n): return {"name": n, "type": "category"}
_PAYMENT_TERM_OPTIONS = ["Immediate Payment", "15 days", "21 days"]
_LINE_WIN_PAYMENT_TERM_OPTIONS = ["15 Days", "21 Days", "Immediate", "Net 30"]

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
        "title": "Smart Quote Optimiser", "model_type": "Classification Model",
        "description": "Enhances quote accuracy by analysing past successes and associated product "
                       "pricing. Helps sales teams win more jobs while maintaining profitability "
                       "with “Win Probability” scores.",
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
        "header_fields": [
            _cat("customerID"), _cat("salesRepID"), _cat("region"),
            _num("leadTimeDays"),
            {"name": "paymentTerms", "type": "category",
             "options": _PAYMENT_TERM_OPTIONS},
        ],
        # listPrice is intentionally absent from the client-facing schema. Live
        # feature building derives it as unitPrice + 67% when an API caller does not
        # supply one, so the model still scores on a list-price basis without asking
        # a salesperson for an unavailable field.
        # negotiatedSalesPrice and materialSpec are deliberately NOT collected on this
        # card. MaXXFlow records a single client-facing price rather than a separate
        # negotiated figure, and materialSpec is product master data a rep should not
        # be retyping. live_features already treats both as optional, so omitting them
        # breaks nothing — product_type just falls back to the unknown level on a
        # champion that happened to be trained with it.
        "line_fields": [_cat("productID"), _num("quantity"), _num("unitPrice"),
                         _num("salesPrice")],
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
        "description": "Forecasts stock-out risks using AI-driven analysis of historical usage "
                       "and supplier behaviour. Enables smarter procurement decisions and reduces "
                       "costly production halts.",
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

# Only these two capabilities are public in the Configurator. Keep the other M1
# definitions private because Smart Quote's bundled prediction still reads the
# existing price champion internally; they must not appear in GET /models or be
# addressable through the generic model endpoints.
_INTERNAL_MODELS = MODELS
MODELS = {
    key: _INTERNAL_MODELS[key]
    for key in ("m1_quote_line_win", "m2_inventory")
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


def _champion_card_metadata(registry: MLflowRegistry, serving_name: str) -> dict:
    """Card metadata belongs to the live champion, never the newest candidate."""
    try:
        tags = registry.get_alias_tags(name=serving_name, alias="champion")
    except Exception:
        tags = {}
    kind = tags.get("data_source_kind") or tags.get("data_provenance")
    kind = "database" if kind == "db" else kind
    if kind not in ("csv", "database"):
        data_source = None
    else:
        raw_count = tags.get("dataset_row_count")
        try:
            row_count = int(raw_count) if raw_count is not None else None
        except (TypeError, ValueError):
            row_count = None
        data_source = {
            "kind": kind,
            "name": tags.get("data_source_name") or (
                "MaXXflow Database" if kind == "database" else "CSV Dataset"
            ),
            "row_count": row_count,
            "fallback_used": str(tags.get("fallback_used", "false")).lower() == "true",
            "fallback_reason": tags.get("fallback_reason"),
        }
    return {
        "algorithm": tags.get("algorithm"),
        "data_source": data_source,
        "published_at": tags.get("published_at"),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/{tenant}/models")
def list_models(tenant: str):
    reg = MLflowRegistry()
    try:
        durable = get_training_result_store().pending(tenant)
    except Exception:
        durable = []
    try:
        live = get_training_backend().pending_runs(tenant)
    except Exception:
        # Model discovery must remain available if the training backend itself is
        # temporarily unreachable. The Train screen will surface its detailed error.
        live = []
    pending_by_model = {}
    for run in sorted([*durable, *live], key=lambda item: float(item.get("started_at") or 0)):
        pending_by_model[run["model_key"]] = run
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
        card_metadata = (_champion_card_metadata(reg, serving_name) if resolved else {
            "algorithm": None, "data_source": None, "published_at": None,
        })
        pending = pending_by_model.get(key)
        if pending is not None and champ is not None:
            try:
                if int((pending.get("result") or {}).get("version")) <= int(champ["version"]):
                    pending = None
            except (KeyError, TypeError, ValueError):
                pass
        run_summary = None if pending is None else {
            "run_id": pending["run_id"],
            "status": pending["status"],
            "progress": pending.get("progress"),
            "elapsed_s": pending.get("elapsed_s", 0),
            "started_at": pending.get("started_at"),
            "data_source": pending.get("data_source"),
        }
        if run_summary is not None and pending["status"] == "done":
            run_summary.update({
                "publication_status": "ready_to_publish",
                "status_label": "Ready to Publish",
                "source": pending.get("source"),
                "finished_at": pending.get("finished_at"),
                "logs": pending.get("logs", []),
                # Everything the Results page and subsequent /publish request need.
                # Training backends already remove the in-memory model object and logs
                # before exposing this result, so the catalogue stays JSON-safe.
                "result": pending.get("result"),
            })
        cards.append({
            "key": key, "title": m["title"], "model_type": m["model_type"],
            "description": m["description"], "predicts": m["predicts"],
            "status": status, "champion": champ,
            "registered_name": own,          # where THIS tenant's training registers
            "serving_name": serving_name,     # what currently answers predictions
            "base_model": is_base,            # True = served by the shared global base
            "training": run_summary if pending and pending["status"] == "running" else None,
            # A registered candidate is not live until Publish moves the @champion alias.
            # Keep this server-derived so leaving the wizard does not erase the state.
            "pending_candidate": run_summary if pending and pending["status"] == "done" else None,
            **card_metadata,
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
        # Explicit business-approved choices (currently paymentTerms) take
        # precedence over whatever spelling happens to be stored in an older
        # champion's category vocabulary.
        if f["type"] == "category" and vocab is not None and not f.get("options"):
            f["options"] = [str(c) for c in vocab][:200]
    return out


def _region_options(tenant: str) -> list[str]:
    """Distinct customer regions/states already on file for this tenant, read live
    from `contacts.state` — there is no dedicated master-data table for region, so
    this beats a champion's frozen training vocab (or an empty box) for a field a
    salesperson expects to pick a real, current value from."""
    try:
        from maxxflow_data.engine import get_data_access

        da = get_data_access()
        if not da.settings.db_enabled:
            return []
        df = da.query(
            "SELECT DISTINCT state FROM contacts "
            "WHERE state IS NOT NULL AND state <> '' AND deleted_at IS NULL "
            "ORDER BY state",
            tenant=_data_tenant(tenant),
        )
        return [str(v) for v in df["state"].tolist()]
    except Exception:
        return []


def _apply_region_options(fields: list[dict], tenant: str) -> list[dict]:
    """Override the `region` category field's options with live tenant data, when
    any exists. Leaves every other field — and `region` itself when the tenant has
    no contacts with a state on file yet — untouched."""
    regions = _region_options(tenant)
    if not regions:
        return fields
    out = []
    for f in fields:
        if f["name"] == "region" and f["type"] == "category":
            f = dict(f)
            f["options"] = regions
        out.append(f)
    return out


def _product_catalog(tenant: str) -> dict[str, dict]:
    """Live tenant products keyed by SKU for Test Predictions.

    The UI-facing identifier is the stable, recognisable SKU. The database UUID is
    retained privately so a DB-trained champion whose productID vocabulary contains UUIDs
    can still receive the representation on which it was trained.
    """
    try:
        from maxxflow_data.engine import get_data_access

        da = get_data_access()
        if not da.settings.db_enabled:
            return {}
        df = da.query(
            "SELECT id, sku, unit_cost, sales_price FROM products "
            "WHERE deleted_at IS NULL ORDER BY sku",
            tenant=_data_tenant(tenant),
        )
        catalog = {}
        for row in df.to_dict(orient="records"):
            sku = str(row.get("sku") or "").strip()
            if sku:
                catalog[sku] = row
        return catalog
    except Exception:
        # Schema discovery must remain usable when the tenant DB is temporarily down.
        return {}


def _line_win_predict_fields(tenant: str) -> list[dict]:
    """The intentionally small, client-facing flat Test Predictions contract."""
    products = list(_product_catalog(tenant))
    fields = [
        _num("quantity"),
        _num("unitPrice"),
        _num("leadTimeDays"),
        _num("salesPrice"),
        {**_num("quote_total"), "read_only": True,
         "formula": "quantity * salesPrice"},
        _num("list_price"),
        {**_cat("productID"), "options": products},
        _cat("region"),
        {**_cat("payment_terms"), "options": _LINE_WIN_PAYMENT_TERM_OPTIONS},
    ]
    return _apply_region_options(fields, tenant)


@app.get("/api/{tenant}/models/{key}/predict-schema")
def predict_schema(tenant: str, key: str):
    m = _model_or_404(key)
    if key == "m2_inventory":
        raise HTTPException(
            404,
            "m2_inventory uses /test-predict for DB input or /predict-schema-csv "
            "for CSV test input",
        )
    if key == "m1_quote_line_win":
        return {"model": key, "fields": _line_win_predict_fields(tenant)}
    fields = _enrich_options(m["fields"], _champion_categories(tenant, key))
    fields = _apply_region_options(fields, tenant)
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


def _line_win_prediction_records(tenant: str, records: list[dict], model) -> list[dict]:
    """Expand the small public line-win form into the champion's engineered contract."""
    catalog = _product_catalog(tenant)
    try:
        categories = getattr(model.unwrap_python_model(), "categories", {}) or {}
    except Exception:
        categories = {}
    trained_products = {str(value) for value in categories.get("productID", [])}
    prepared = []
    for index, original in enumerate(records):
        record = dict(original)
        product_sku = str(record.get("productID") or "").strip()
        product = catalog.get(product_sku, {})

        # Accept the established camelCase name and the lowercase spelling used by
        # older/manual API callers.
        sales_price = record.get("salesPrice", record.get("salesprice"))
        unit_price = record.get("unitPrice", product.get("unit_cost"))
        if sales_price in (None, ""):
            sales_price = product.get("sales_price")
        list_price = record.get("list_price", record.get("listPrice"))
        if list_price in (None, ""):
            list_price = product.get("sales_price")
        try:
            quantity = float(record.get("quantity"))
            sales_price = float(sales_price)
            unit_price = float(unit_price)
        except (TypeError, ValueError):
            raise HTTPException(
                400,
                f"record {index}: quantity, unitPrice and salesPrice must be numeric",
            )
        try:
            resolved_list_price = float(list_price)
        except (TypeError, ValueError):
            resolved_list_price = unit_price

        # DB training learns the product UUID; CSV training commonly learns the SKU.
        # Send whichever form exists in the active champion's vocabulary.
        db_product_id = str(product.get("id") or "")
        if product_sku not in trained_products and db_product_id in trained_products:
            record["productID"] = db_product_id
        else:
            record["productID"] = product_sku

        ratio, _ = price_ratio_for(sales_price, resolved_list_price, unit_price)
        record.update({
            "quantity": quantity,
            "unitPrice": unit_price,
            "salesPrice": sales_price,
            "list_price": resolved_list_price,
            "price_ratio": ratio,
            "quote_total": quantity * sales_price,
            # These engineered fields are deliberately hidden from the public schema.
            # Neutral values satisfy older required MLflow signatures; the served model
            # handles unknown categories and applies its own learned numeric fills.
            "contact_win_rate": 0.5,
            "salesrep_win_rate": 0.5,
            "industry": "",
            "product_type": "",
        })
        prepared.append(record)
    return prepared


@app.post("/api/{tenant}/models/{key}/predict")
def predict(tenant: str, key: str, payload: dict):
    m = _model_or_404(key)
    if key == "m2_inventory":
        raise HTTPException(
            404,
            "m2_inventory uses /test-predict for DB input or /predict-csv for CSV input",
        )
    records = payload.get("records") or [payload]

    reg = MLflowRegistry()
    loaded = _load_champion(reg, tenant, key)
    if loaded is None:
        raise HTTPException(409, "no published (champion) model yet, and no global base model "
                                 "to fall back to — train and publish first")
    serving_name, is_base, model = loaded
    if key == "m1_quote_line_win":
        records = _line_win_prediction_records(tenant, records, model)
    df = _coerce(pd.DataFrame(records), m["fields"])

    preds = model.predict(df)

    # pandas' JSON encoder converts guardrail NaN values to JSON null. Returning
    # to_dict() directly makes Starlette reject an otherwise valid prediction with
    # "Out of range float values are not JSON compliant".
    predictions = json.loads(preds.to_json(orient="records"))
    return {"predictions": predictions,
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
    # The form exposes recognisable product SKUs from the tenant DB. A champion
    # trained from that DB may have learned the product UUID FK instead; translate
    # only when the UUID, rather than the SKU, is present in its vocabulary.
    try:
        win_categories = getattr(win_model.unwrap_python_model(), "categories", {}) or {}
    except Exception:
        win_categories = {}
    trained_product_ids = {str(value) for value in win_categories.get("productID", [])}
    product_catalog = _product_catalog(tenant)
    for raw_line, win_record in zip(lines, win_records):
        sku = str(raw_line.get("productID") or "").strip()
        db_id = str(product_catalog.get(sku, {}).get("id") or "")
        if sku not in trained_product_ids and db_id in trained_product_ids:
            win_record["productID"] = db_id
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
        price_df = _coerce(pd.DataFrame(price_records),
                           _INTERNAL_MODELS["m1_quote_price"]["fields"])
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
    mil_df = _coerce(pd.DataFrame(mil_records), _INTERNAL_MODELS["m1_quote_mil"]["fields"])
    mil_preds = mil_model.predict(mil_df).to_dict(orient="records")

    price_loaded = _load_champion(reg, tenant, "m1_quote_price")
    price_preds = None
    if price_loaded is not None:
        _, _, price_model = price_loaded
        price_df = _coerce(pd.DataFrame(price_records),
                           _INTERNAL_MODELS["m1_quote_price"]["fields"])
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


def _line_win_raw_payload(payload: dict, tenant: str) -> tuple[dict, list[dict]]:
    """Accept both the canonical raw shape and the Configurator's flat form shape.

    Canonical callers send ``{header, lines}``. The flat Test Predictions schema sends
    ``{records: [...]}``; split those records here so both paths reach the same feature
    derivation and the same model bundle.
    """
    if "records" not in payload:
        return payload.get("header") or {}, payload.get("lines") or []

    records = payload.get("records") or []
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise HTTPException(400, "records must be a list of objects")
    if not records:
        return {}, []

    first = records[0]
    header = {
        "customerID": first.get("customerID"),
        "salesRepID": first.get("salesRepID"),
        "region": first.get("region", ""),
        "leadTimeDays": first.get("leadTimeDays", 0),
        "paymentTerms": first.get("payment_terms", first.get("paymentTerms")),
    }
    catalog = _product_catalog(tenant)
    lines = []
    for record in records:
        sku = str(record.get("productID") or "").strip()
        product = catalog.get(sku, {})
        unit_price = record.get("unitPrice")
        if unit_price in (None, ""):
            unit_price = product.get("unit_cost")
        sales_price = record.get("salesPrice", record.get("salesprice"))
        if sales_price in (None, ""):
            sales_price = product.get("sales_price")
        list_price = record.get("list_price", record.get("listPrice"))
        if list_price in (None, ""):
            list_price = product.get("sales_price")
        lines.append({
            "productID": sku,
            "quantity": record.get("quantity"),
            "unitPrice": unit_price,
            "salesPrice": sales_price,
            "listPrice": list_price,
        })
    return header, lines


@app.get("/api/{tenant}/models/{key}/predict-schema-raw")
def predict_schema_raw(tenant: str, key: str):
    """Raw, client-facing test-input columns — what a salesperson actually has on
    hand (customer, region, lead time, product lines), never a computed ratio or
    win-rate."""
    m = _model_or_404(key)
    if "header_fields" not in m:
        raise HTTPException(400, f"{key} has no raw test-input schema yet")
    cats, graph = _champion_categories(tenant, key), _load_option_graph()
    if key == "m1_quote_line_win":
        # productID used to cascade from industry. Industry is no longer a Test
        # Prediction input, so do not return a dependency on a field that the form
        # cannot render; the full product vocabulary remains available.
        graph = {name: spec for name, spec in graph.items() if name != "productID"}
    header_fields = _apply_option_graph(_enrich_options(m["header_fields"], cats), graph)
    header_fields = _apply_region_options(header_fields, tenant)
    return {"model": key,
            "header_fields": header_fields,
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
    if key == "m1_quote_line_win":
        header, lines = _line_win_raw_payload(payload, tenant)
    else:
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


def _preview_page(items: Sequence, page: int, page_size: int) -> tuple[list, dict]:
    """Return one 1-based page plus metadata shared by both preview collections."""
    total_items = len(items)
    total_pages = math.ceil(total_items / page_size) if total_items else 0
    start = (page - 1) * page_size
    return list(items[start:start + page_size]), {
        "page": page,
        "page_size": page_size,
        "total_items": total_items,
        "total_pages": total_pages,
        "has_previous": page > 1 and total_items > 0,
        "has_next": page < total_pages,
    }


def _column_search(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise HTTPException(400, "column_search must be a string")
    search = value.strip()
    if len(search) > 100:
        raise HTTPException(400, "column_search must be 100 characters or fewer")
    return search


def _sample_search(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise HTTPException(400, "sample_search must be a string")
    search = value.strip()
    if len(search) > 100:
        raise HTTPException(400, "sample_search must be 100 characters or fewer")
    return search


def _sample_preview_page_size(value: int) -> int:
    if value not in _SAMPLE_PREVIEW_PAGE_SIZES:
        allowed = ", ".join(str(size) for size in _SAMPLE_PREVIEW_PAGE_SIZES)
        raise HTTPException(400, f"sample_page_size must be one of: {allowed}")
    return value


def _search_columns(columns: Sequence[str], search: str) -> list[str]:
    if not search:
        return list(columns)
    needle = search.casefold()
    return [column for column in columns if needle in column.casefold()]


def _search_sample(df: pd.DataFrame, search: str) -> pd.DataFrame:
    if not search or df.empty:
        return df
    matches = pd.Series(False, index=df.index)
    for column in df.columns:
        matches |= df[column].astype("string").str.contains(
            search, case=False, regex=False, na=False,
        )
    return df.loc[matches]


def _unique_value_count(series: pd.Series) -> int:
    """Distinct non-null values in the full column, not merely the sample page."""
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        # Defensive fallback for object columns containing lists/dicts. CSV and the
        # current DB frames are scalar, but preview metadata should remain JSON-safe
        # when a future feature builder carries a structured value.
        normalised = series.dropna().map(
            lambda value: json.dumps(value, sort_keys=True, default=str)
        )
        return int(normalised.nunique(dropna=True))


def _column_metadata(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    return [
        {"name": column, "unique_value_count": _unique_value_count(df[column])}
        for column in columns
    ]


def _db_column_metadata(key: str, df: pd.DataFrame, columns: list[str]) -> list[dict]:
    if key == "m2_inventory":
        from m2_inventory.db_training import column_metadata
        metadata = column_metadata(df, columns)
    else:
        from m1_quote import db_features
        metadata = db_features.column_metadata(key, columns)
    counts = {item["name"]: item["unique_value_count"]
              for item in _column_metadata(df, columns)}
    return [{**item, "unique_value_count": counts[item["name"]]} for item in metadata]


def _retrain_preview_response(
    *,
    key: str,
    upload_id: str,
    filename: str,
    df: pd.DataFrame,
    columns_page: int,
    columns_page_size: int,
    sample_page: int,
    sample_page_size: int,
    column_search: str = "",
    sample_search: str = "",
) -> dict:
    """Build the validation result with independently paginated columns and rows."""
    m = _model_or_404(key)
    column_search = _column_search(column_search)
    sample_search = _sample_search(sample_search)
    sample_page_size = _sample_preview_page_size(sample_page_size)
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
    all_columns = list(df.columns)
    matching_columns = _search_columns(all_columns, column_search)
    columns, columns_pagination = _preview_page(
        matching_columns, columns_page, columns_page_size,
    )
    # Convert through pandas' JSON encoder so NaN becomes JSON null. Slicing before
    # conversion also avoids serialising the complete uploaded dataset for every page.
    matching_sample = _search_sample(df, sample_search)
    sample_start = (sample_page - 1) * sample_page_size
    sample = json.loads(
        matching_sample.iloc[sample_start:sample_start + sample_page_size].to_json(
            orient="records"
        )
    )
    _, sample_pagination = _preview_page(
        range(len(matching_sample)), sample_page, sample_page_size,
    )

    return {
        "upload_id": upload_id, "filename": filename,
        "rows": int(len(df)), "columns": columns,
        "column_metadata": _column_metadata(df, columns),
        "required_columns": m["required"], "missing_columns": missing,
        "optional_columns": optional, "optional_present": opt_present,
        "optional_missing": opt_missing,
        "valid": not missing,
        "message": msg,
        "sample": sample,
        "search": {
            "column_search": column_search,
            "matched_columns": len(matching_columns),
            "total_columns": len(all_columns),
            "sample_search": sample_search,
            "matched_rows": len(matching_sample),
            "total_rows": len(df),
        },
        "pagination": {
            "columns": columns_pagination,
            "sample": {
                **sample_pagination,
                "allowed_page_sizes": list(_SAMPLE_PREVIEW_PAGE_SIZES),
            },
        },
    }


def _load_preview_upload(upload_id: str) -> tuple[Path, str, int | None]:
    """Resolve a previously uploaded preview without allowing path traversal."""
    if len(upload_id) != 12 or any(c not in "0123456789abcdef" for c in upload_id):
        raise HTTPException(404, "upload not found — re-run preview")
    path = _UPLOADS / f"{upload_id}.csv"
    if not path.exists():
        raise HTTPException(404, "upload not found — re-run preview")
    metadata_path = _UPLOADS / f"{upload_id}.json"
    try:
        metadata = json.loads(metadata_path.read_text())
        filename = str(metadata.get("filename") or path.name)
        rows = int(metadata["rows"]) if metadata.get("rows") is not None else None
    except (OSError, ValueError, TypeError):
        filename = path.name
        rows = None
    return path, filename, rows


@app.post("/api/{tenant}/models/{key}/retrain/preview")
async def retrain_preview(
    tenant: str,
    key: str,
    file: UploadFile = File(...),
    columns_page: int = Query(1, ge=1),
    columns_page_size: int = Query(_COLUMN_PREVIEW_PAGE_SIZE, ge=1, le=100),
    sample_page: int = Query(1, ge=1),
    sample_page_size: int = Query(5),
    column_search: str = Query("", max_length=100),
    sample_search: str = Query("", max_length=100),
):
    _model_or_404(key)
    upload_id = uuid.uuid4().hex[:12]
    path = _UPLOADS / f"{upload_id}.csv"
    path.write_bytes(await file.read())
    filename = file.filename or path.name
    df = pd.read_csv(path)
    (_UPLOADS / f"{upload_id}.json").write_text(json.dumps({
        "filename": filename,
        "rows": int(len(df)),
    }))
    return _retrain_preview_response(
        key=key,
        upload_id=upload_id,
        filename=filename,
        df=df,
        columns_page=columns_page,
        columns_page_size=columns_page_size,
        sample_page=sample_page,
        sample_page_size=sample_page_size,
        column_search=_column_search(column_search),
        sample_search=_sample_search(sample_search),
    )


@app.get("/api/{tenant}/models/{key}/retrain/preview/{upload_id}")
def retrain_preview_page(
    tenant: str,
    key: str,
    upload_id: str,
    columns_page: int = Query(1, ge=1),
    columns_page_size: int = Query(_COLUMN_PREVIEW_PAGE_SIZE, ge=1, le=100),
    sample_page: int = Query(1, ge=1),
    sample_page_size: int = Query(5),
    column_search: str = Query("", max_length=100),
    sample_search: str = Query("", max_length=100),
):
    """Read another preview page without uploading the same dataset again."""
    _model_or_404(key)
    path, filename, _ = _load_preview_upload(upload_id)
    return _retrain_preview_response(
        key=key,
        upload_id=upload_id,
        filename=filename,
        df=pd.read_csv(path),
        columns_page=columns_page,
        columns_page_size=columns_page_size,
        sample_page=sample_page,
        sample_page_size=sample_page_size,
        column_search=_column_search(column_search),
        sample_search=_sample_search(sample_search),
    )


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
    dt = _data_tenant(tenant)
    if key == "m2_inventory":
        from m2_inventory.db_training import describe_sources

        note = (
            "M2 uses real weekly history when available. Otherwise it builds a marked "
            "bootstrap fallback from current operational inventory so training can complete."
        )
        try:
            rep = describe_sources(dt)
            rep.update({"model": key, "error": None, "note": note, "data_tenant": dt})
            return rep
        except Exception as e:
            return {"model": key, "connected": False,
                    "error": f"read replica not reachable ({type(e).__name__}: {e})",
                    "schema": None, "sources": [], "data_tenant": dt,
                    "passthrough_columns": [], "derived_features": [],
                    "trainable": False, "note": note}

    from m1_quote import db_features
    note = ("The DB path reads these source tables from the tenant read replica and builds the "
            "gold features at training time; the listed features are DERIVED (never physical "
            "columns). Train with POST /train {\"source\":\"db\"}.")
    spec = db_features._SOURCE_SPEC.get(key)
    if spec is None:
        return {"model": key, "connected": False, "error": "no DB source spec for this model",
                "schema": None, "sources": [], "passthrough_columns": [],
                "derived_features": [], "trainable": False, "note": note}
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
    with independently paginated columns and sample rows.

    Errors come back 200 with ``ok: false``, matching db-columns, so the wizard
    renders a banner instead of a network failure.
    """
    _model_or_404(key)
    payload = payload or {}
    column_search = _column_search(payload.get("column_search"))
    sample_search = _sample_search(payload.get("sample_search"))
    try:
        columns_page = int(payload.get("columns_page", 1))
        sample_page = int(payload.get("sample_page", 1))
        sample_page_size = int(
            payload.get("sample_page_size", payload.get("limit", 5))
        )
    except (TypeError, ValueError):
        raise HTTPException(400, "pagination values must be integers")
    if columns_page < 1 or sample_page < 1:
        raise HTTPException(400, "pagination pages must be at least 1")
    if sample_page_size not in _SAMPLE_PREVIEW_PAGE_SIZES:
        allowed = ", ".join(str(size) for size in _SAMPLE_PREVIEW_PAGE_SIZES)
        raise HTTPException(400, f"sample_page_size must be one of: {allowed}")

    if key not in jobs.db_models():
        return {"ok": False, "model": key, "tenant": tenant,
                "error": f"{key} has no DB-path builder — it is CSV-upload only. "
                         f"Models with a DB path: {', '.join(jobs.db_models())}.",
                "rows": 0, "columns": [], "column_metadata": [],
                "dtypes": {}, "sample": []}

    dt = _data_tenant(tenant)
    t0 = time.perf_counter()
    try:
        df = jobs.build_db_frame(dt, key)
    except Exception as e:
        return {"ok": False, "model": key, "tenant": tenant, "data_tenant": dt,
                "error": f"{type(e).__name__}: {e}",
                "rows": 0, "columns": [], "column_metadata": [],
                "dtypes": {}, "sample": []}

    rows, all_columns = int(len(df)), list(df.columns)
    matching_columns = _search_columns(all_columns, column_search)
    columns, columns_pagination = _preview_page(
        matching_columns, columns_page, _COLUMN_PREVIEW_PAGE_SIZE,
    )
    matching_sample = _search_sample(df, sample_search)
    sample_start = (sample_page - 1) * sample_page_size
    sample = json.loads(
        matching_sample.iloc[sample_start:sample_start + sample_page_size].to_json(
            orient="records"
        )
    )
    _, sample_pagination = _preview_page(
        range(len(matching_sample)), sample_page, sample_page_size,
    )
    # Empty is not an error — it is the single most useful thing this endpoint
    # can tell you, and it is invisible from a row count on the source tables.
    if rows == 0:
        message = (f"The query ran against tenant_{dt} and returned no training rows."
                   if key == "m2_inventory" else
                   f"The query ran against tenant_{dt} and returned no rows. The source tables "
                   "have data, so the loss is in the join or the label filter — most often "
                   "quotations whose stage/status never resolved to won or lost.")
    else:
        message = (f"{rows} rows x {len(all_columns)} columns built from schema "
                   f"tenant_{dt}.")

    return {
        "ok": True, "model": key, "tenant": tenant, "data_tenant": dt,
        "rows": rows, "columns": columns,
        "column_metadata": _db_column_metadata(key, df, columns),
        "dtypes": {c: str(df.dtypes[c]) for c in columns},
        "sample": sample,
        "sample_size": len(sample),
        "truncated": sample_pagination["total_pages"] > 1,
        "search": {
            "column_search": column_search,
            "matched_columns": len(matching_columns),
            "total_columns": len(all_columns),
            "sample_search": sample_search,
            "matched_rows": len(matching_sample),
            "total_rows": rows,
        },
        "pagination": {
            "columns": columns_pagination,
            "sample": {
                **sample_pagination,
                "allowed_page_sizes": list(_SAMPLE_PREVIEW_PAGE_SIZES),
            },
        },
        "data_source": {
            "kind": "database",
            "name": df.attrs.get("source_name", "MaXXflow Database"),
            "row_count": rows,
            "fallback_used": bool(df.attrs.get("fallback_used")),
            "fallback_reason": df.attrs.get("fallback_reason"),
        },
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
                                              auto_hpo=auto_hpo, data_tenant=dt,
                                              source_name="MaXXflow Database",
                                              dataset_row_count=None)
        return {"run_id": run_id, "source": "db", "data_tenant": dt}
    upload_id = payload.get("upload_id")     # Upload-a-File path
    if not upload_id:
        raise HTTPException(400, "provide upload_id (from /retrain/preview) or source='db'")
    path, filename, row_count = _load_preview_upload(str(upload_id))
    # Compatibility for previews created before row counts were added to their
    # sidecar. New uploads do not need this second read.
    if row_count is None:
        row_count = int(len(pd.read_csv(path)))
    run_id = get_training_backend().start(tenant, key, source="csv", csv_path=str(path),
                                          auto_hpo=auto_hpo, source_name=filename,
                                          dataset_row_count=row_count)
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
    # `force` is an explicit human decision to ship despite failed performance
    # checks (absolute floor or champion comparison). It never overrides invalid
    # input data, and every failed check remains in the returned audit trail.
    force = payload.get("force", False)
    if not isinstance(force, bool):
        raise HTTPException(400, "force must be a JSON boolean")
    result = m["trainer"].publish(tenant, str(version), metrics, force=force)
    result["force_requested"] = force
    result["forced"] = bool(
        force and result.get("published")
        and any(not check.get("passed", False) for check in result.get("gate_checks", []))
    )
    if result.get("published"):
        published_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        MLflowRegistry().set_version_tags(
            name=registered_model_name(tenant, key),
            version=str(version),
            tags={"published_at": published_at},
        )
        result["published_at"] = published_at
        get_training_backend().mark_published(tenant, key, str(version))
        try:
            get_training_result_store().mark_published(tenant, key, str(version))
        except Exception:
            result["result_store_warning"] = (
                "Model published, but the durable training-result status could not be updated."
            )
    return result


@app.get("/api/{tenant}/inventory-dashboard")
def inventory_dashboard(
    tenant: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(10),
):
    """Bulk-score current tenant inventory and return one risk-ranked page."""
    if page_size not in _INVENTORY_DASHBOARD_PAGE_SIZES:
        allowed = ", ".join(str(size) for size in _INVENTORY_DASHBOARD_PAGE_SIZES)
        raise HTTPException(400, f"page_size must be one of: {allowed}")

    try:
        snapshots = read_db_prediction_snapshots(tenant, get_clock().as_of())
        if snapshots.empty:
            raise ValueError("the tenant database returned no inventory items")
        latest = (
            snapshots.sort_values("snapshot_date")
            .groupby(["item_id", "warehouse_id"], as_index=False, sort=False)
            .tail(1)
            .reset_index(drop=True)
        )
        risks, algorithm = inventory_csv._score_via_champion(
            tenant, inventory_csv._coerce_model_input(latest)
        )
        model_source = "published champion (Data Lake artifact)"
    except Exception as exc:
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
            "warehouse_id": str(row.warehouse_id),
            "item_name": str(getattr(row, "item_name", "") or row.item_id),
            "product": str(getattr(row, "item_name", "") or row.item_id),
            "part_number": str(getattr(row, "part_number", "") or ""),
            "warehouse": str(getattr(row, "warehouse_name", "") or row.warehouse_id),
            "quantity": available,
            "reserved_quantity": reserved,
            "available_quantity": available,
            "net_available_quantity": available - reserved,
            "rop": number(row.rop),
            "reorder_point": number(row.rop),
            "forecast_quantity": number(row.demand_forecast_qty),
            "risk_30d": float(row.risk_30d),
            "risk_60d": float(row.risk_60d),
            "badge_30d": str(row.badge_30d),
            "badge_60d": str(row.badge_60d),
            "suppressed": bool(row.suppressed),
            "model_version": str(row.model_version),
        })
    paged_rows, pagination = _preview_page(rows, page, page_size)
    return {
        "tenant": tenant,
        "snapshot_date": latest["snapshot_date"].max().date().isoformat(),
        "model_source": model_source,
        "selected_model": algorithm,
        "model_metrics": [],
        "selection_warning": None,
        "rows": paged_rows,
        "pagination": {
            **pagination,
            "allowed_page_sizes": list(_INVENTORY_DASHBOARD_PAGE_SIZES),
        },
        "summary": {
            "products": len(rows),
            "high_risk_30d": sum(row["risk_30d"] >= 0.66 for row in rows),
            "medium_risk_30d": sum(0.33 <= row["risk_30d"] < 0.66 for row in rows),
            "low_risk_30d": sum(row["risk_30d"] < 0.33 for row in rows),
        },
    }


_INVENTORY_TEST_OVERRIDES = {
    "available_qty", "reserved_qty", "forecasted_qty", "rop",
    "demand_forecast_qty", "past_due_qty", "open_qty", "lead_time_days",
}
_INVENTORY_TEST_REQUIRED = (
    "available_qty", "reserved_qty", "rop", "demand_forecast_qty",
)


def _inventory_test_value(row: pd.Series, field: str):
    value = row.get(field)
    return None if pd.isna(value) else float(value)


def _inventory_test_text(value, fallback: str = "") -> str:
    return fallback if value is None or pd.isna(value) else str(value)


def _inventory_selected_input(row: pd.Series) -> dict:
    """Public, JSON-safe DB values used to populate the single-test form."""
    return {
        "item_id": str(row["item_id"]),
        "warehouse_id": str(row["warehouse_id"]),
        "item_name": _inventory_test_text(row.get("item_name"), str(row["item_id"])),
        "part_number": _inventory_test_text(row.get("part_number")),
        "warehouse": _inventory_test_text(
            row.get("warehouse_name"), str(row["warehouse_id"])
        ),
        "snapshot_date": pd.Timestamp(row["snapshot_date"]).date().isoformat(),
        **{
            field: _inventory_test_value(row, field)
            for field in sorted(_INVENTORY_TEST_OVERRIDES)
        },
    }


@app.get("/api/{tenant}/models/m2_inventory/test-predict-schema")
def inventory_test_predict_schema(
    tenant: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=100),
    search: str = Query("", max_length=100),
    item_id: str | None = Query(None),
    warehouse_id: str | None = Query(None),
):
    """Paginate DB items and optionally load one location's prediction inputs."""
    try:
        snapshots = read_db_prediction_snapshots(tenant, get_clock().as_of())
    except Exception as exc:
        raise HTTPException(409, f"inventory prediction data is not ready: {exc}") from exc

    selected = None
    requested_item = str(item_id or "").strip()
    requested_warehouse = str(warehouse_id or "").strip()
    if bool(requested_item) != bool(requested_warehouse):
        raise HTTPException(400, "item_id and warehouse_id must be provided together")
    if requested_item:
        match = snapshots[
            snapshots["item_id"].astype(str).eq(requested_item)
            & snapshots["warehouse_id"].astype(str).eq(requested_warehouse)
        ]
        if match.empty:
            raise HTTPException(404, "item and warehouse location were not found")
        selected = _inventory_selected_input(match.iloc[0])

    filtered = snapshots
    needle = search.strip().casefold()
    if needle and not filtered.empty:
        matches = pd.Series(False, index=filtered.index)
        for column in (
            "item_id", "item_name", "part_number", "warehouse_id", "warehouse_name",
        ):
            if column in filtered:
                matches |= filtered[column].astype("string").str.contains(
                    needle, case=False, regex=False, na=False,
                )
        filtered = filtered.loc[matches]

    total_items = int(len(filtered))
    total_pages = math.ceil(total_items / page_size) if total_items else 0
    start = (page - 1) * page_size
    items = [
        {
            "item_id": str(row["item_id"]),
            "warehouse_id": str(row["warehouse_id"]),
            "item_name": _inventory_test_text(row.get("item_name"), str(row["item_id"])),
            "part_number": _inventory_test_text(row.get("part_number")),
            "warehouse": _inventory_test_text(
                row.get("warehouse_name"), str(row["warehouse_id"])
            ),
        }
        for _, row in filtered.iloc[start:start + page_size].iterrows()
    ]
    return {
        "model": "m2_inventory",
        "tenant": tenant,
        "fields": [
            {"name": "item_id", "type": "category", "required": True},
            {"name": "warehouse_id", "type": "category", "required": True},
            {"name": "snapshot_date", "type": "date", "read_only": True},
            *[
                {
                    "name": field,
                    "type": "number",
                    "required": field in _INVENTORY_TEST_REQUIRED,
                }
                for field in sorted(_INVENTORY_TEST_OVERRIDES)
            ],
        ],
        "items": items,
        "selected": selected,
        "search": search.strip(),
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages,
            "has_previous": page > 1 and total_items > 0,
            "has_next": page < total_pages,
        },
    }


@app.post("/api/{tenant}/models/m2_inventory/test-predict")
def inventory_test_predict(tenant: str, payload: dict):
    """Score one current item/location, reading its hidden feature context from DB."""
    record = payload.get("record", payload)
    if not isinstance(record, dict):
        raise HTTPException(400, "provide a JSON object in 'record'")
    item_id = str(record.get("item_id") or "").strip()
    warehouse_id = str(record.get("warehouse_id") or "").strip()
    if not item_id or not warehouse_id:
        raise HTTPException(400, "item_id and warehouse_id are required")

    as_of = get_clock().as_of()
    try:
        snapshots = read_db_prediction_snapshots(tenant, as_of)
        selected = snapshots[
            snapshots["item_id"].astype(str).eq(item_id)
            & snapshots["warehouse_id"].astype(str).eq(warehouse_id)
        ].copy()
        if selected.empty:
            raise HTTPException(404, "item and warehouse location were not found")

        overrides = record.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise HTTPException(400, "overrides must be a JSON object")
        # Also accept the editable fields at the top level for a simpler form payload.
        overrides = {
            field: record[field]
            for field in _INVENTORY_TEST_OVERRIDES
            if field in record
        } | overrides
        unknown = sorted(set(overrides) - _INVENTORY_TEST_OVERRIDES)
        if unknown:
            raise HTTPException(400, f"unsupported overrides: {unknown}")
        for field, value in overrides.items():
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise HTTPException(400, f"{field} must be numeric") from exc
            if number < 0:
                raise HTTPException(400, f"{field} must not be negative")
            selected.loc[:, field] = number

        predictions, algorithm = inventory_csv._score_via_champion(
            tenant, inventory_csv._coerce_model_input(selected)
        )
        prediction = predictions.iloc[0]
        source = selected.iloc[0]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(409, f"inventory prediction is not ready: {exc}") from exc

    return {
        "tenant": tenant,
        "model": "m2_inventory",
        "snapshot_date": pd.Timestamp(source["snapshot_date"]).date().isoformat(),
        "algorithm": algorithm,
        "model_source": "published champion (Data Lake artifact)",
        "input": {
            "item_id": item_id,
            "warehouse_id": warehouse_id,
            **{
                field: _inventory_test_value(source, field)
                for field in sorted(_INVENTORY_TEST_OVERRIDES)
            },
        },
        "prediction": {
            "item_id": str(prediction.get("item_id", item_id)),
            "warehouse_id": str(prediction.get("warehouse_id", warehouse_id)),
            "risk_30d": float(prediction["risk_30d"]),
            "risk_60d": float(prediction["risk_60d"]),
            "badge_30d": str(prediction["badge_30d"]),
            "badge_60d": str(prediction["badge_60d"]),
            "is_rule_based": bool(prediction["is_rule_based"]),
            "suppressed": bool(prediction["suppressed"]),
            "model_version": str(prediction["model_version"]),
        },
    }


@app.post("/api/{tenant}/models/m2_inventory/batch-predict")
def inventory_batch_predict(tenant: str):
    """Score all current tenant inventory and persist each risk in PostgreSQL."""
    as_of = get_clock().as_of()
    try:
        scored_items = inventory_batch.score(tenant)
    except Exception as exc:
        raise HTTPException(409, f"inventory batch prediction is not ready: {exc}") from exc
    return {
        "tenant": tenant,
        "model": "m2_inventory",
        "status": "completed",
        "snapshot_date": pd.Timestamp(as_of).date().isoformat(),
        "scored_items": scored_items,
        "saved_to": "items.custom_elements.ai_stockout",
        "dashboard_endpoint": f"/api/{tenant}/inventory-dashboard",
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
        model_input = inventory_csv._coerce_model_input(raw)
        summary_path = _INVENTORY_ARTIFACTS / "training_summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists() else {}
        )
        if summary.get("quality_floor_passed", False):
            model = inventory_csv.load_selected_model(_INVENTORY_ARTIFACTS)
            prediction = model.predict_risk(model_input).iloc[0]
            algorithm = model.algorithm_name
            model_source = "latest CSV evaluation winner"
        else:
            # The last comparison run's best-of-4 failed the quality floor —
            # do not silently serve it. Fall back to the published champion,
            # the same safety net `publish` already enforces at promotion time.
            predictions, algorithm = inventory_csv._score_via_champion(tenant, model_input)
            prediction = predictions.iloc[0]
            model_source = "published champion (latest CSV run failed the quality floor)"
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(409, f"inventory prediction is not ready: {exc}") from exc
    return {
        "tenant": tenant,
        "algorithm": algorithm,
        "model_source": model_source,
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
# ---------------------------------------------------------------------------
# M3 — Production Delay insights.
#
# Not a Configurator "model card": M3 trains nothing, registers nothing and has
# no champion, so the generic /predict and /train routes do not apply to it and
# it is deliberately absent from MODELS. Its shape is M2's batch-predict — read
# the tenant, score, review, cache the result on each entity — which is why it
# lives here as its own pair of routes instead.
# ---------------------------------------------------------------------------

M3_ADVISORY_KEY = "ai_delay_insight"


@app.post("/api/{tenant}/models/m3_production_delay/batch-review")
def delay_batch_review(tenant: str, payload: dict | None = None):
    """Score every job's delay risk, review the explanation, cache it on the MO.

    Body (all optional except ``threshold``)::

        {"threshold": 1.0, "jobs": ["WH/MO/00142"], "dry_run": false}

    ``threshold`` is required and has no default anywhere in M3: the composite
    risk score is a weighted mean of raw ratios, so the cutoff that means
    "delayed" is a tenant calibration decision, not something this service can
    pick. Passing one that did not produce the scores is refused rather than
    silently re-badging them.

    ``dry_run`` builds every insight and returns it without writing — the way
    to see what would land on the MOs before it does.
    """
    from m3_production_delay.review.pipeline import run as run_delay_review

    payload = payload or {}
    threshold = payload.get("threshold")
    if threshold is None:
        raise HTTPException(
            422,
            "threshold is required: there is no calibrated default delay cutoff, and the same "
            "composite score means different things for different tenants",
        )
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        raise HTTPException(422, f"threshold must be a number, got {threshold!r}") from None
    if not math.isfinite(threshold):
        raise HTTPException(422, "threshold must be a finite number")

    jobs = payload.get("jobs") or payload.get("job_references")
    if jobs is not None and not isinstance(jobs, list):
        raise HTTPException(422, "jobs must be a list of job references")
    dry_run = bool(payload.get("dry_run", False))

    try:
        insights = run_delay_review(
            tenant=tenant, threshold=threshold, job_references=jobs, dry_run=dry_run
        )
    except ValueError as exc:
        # The threshold guard and the "this is not a scored job" guard both land
        # here: caller errors, not server faults.
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(409, f"delay review is not ready: {exc}") from exc

    by_status: dict[str, int] = {}
    for insight in insights:
        by_status[insight.status] = by_status.get(insight.status, 0) + 1

    return {
        "tenant": tenant,
        "model": "m3_production_delay",
        "status": "completed",
        "dry_run": dry_run,
        "threshold": threshold,
        "reviewed_jobs": len(insights),
        "by_status": by_status,
        "saved_to": (
            None if dry_run else f"manufacturing_orders.custom_elements.{M3_ADVISORY_KEY}"
        ),
        "insights": [insight.to_dict() for insight in insights],
    }


@app.get("/api/{tenant}/delay-insights")
def delay_insights(request: Request, tenant: str, job: str | None = Query(default=None)):
    """Read back the cached insights — what the MO "AI Insights" panel renders.

    This is a plain JSONB read, no model call and no LLM: exactly what the page
    does, which is the point of writing the advisory back in the first place.
    ``?job=WH/MO/00142`` narrows it to one manufacturing order.
    """
    from maxxflow_data.engine import get_data_access

    # Belt and braces. `security._TENANT_PATH` already refuses a path tenant
    # that differs from the authenticated one, but that guard is a regex
    # enumerating path prefixes: a future rename of this route would silently
    # fall out of it and turn this into a cross-tenant read. The query below
    # runs against the PATH tenant, so the route re-checks it itself.
    auth: AuthContext = request.state.auth
    if tenant != auth.tenant_slug:
        raise HTTPException(403, "Path tenant does not match x-tenant-slug")

    sql = (
        "SELECT reference, custom_elements -> :key AS insight "
        "FROM manufacturing_orders "
        "WHERE deleted_at IS NULL AND custom_elements ? :key"
    )
    params: dict = {"key": M3_ADVISORY_KEY}
    if job:
        sql += " AND reference = :reference"
        params["reference"] = job
    sql += " ORDER BY reference"

    try:
        frame = get_data_access().query(sql, params, tenant=tenant)
    except Exception as exc:
        raise HTTPException(409, f"cannot read delay insights: {exc}") from exc

    rows = [
        {"job_id": record["reference"], "insight": record["insight"]}
        for record in frame.to_dict(orient="records")
    ]
    if job and not rows:
        raise HTTPException(
            404,
            f"no cached delay insight for {job!r} — run POST "
            f"/api/{tenant}/models/m3_production_delay/batch-review first",
        )
    return {"tenant": tenant, "count": len(rows), "insights": rows}


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
