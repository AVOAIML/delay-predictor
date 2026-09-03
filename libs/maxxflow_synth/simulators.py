"""Domain simulators (plan §4, §5). Each embeds its module's latent ground-truth
and emits a :class:`ModuleBatch`: schema-faithful raw ``tables`` (for the tenant
Postgres load) plus the ground-truth ``features`` + ``label`` (for the gates and
training). M1 is the reference and builds full raw tables; M2/M3/M4 emit their
feature/label contracts here and gain full raw tables with their modules (§10
checkpoint 3 / 4).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from maxxflow_core.clock import get_clock
from maxxflow_synth import ground_truth as GT
from maxxflow_synth.masterdata_seed import build_masterdata

_DAY = np.timedelta64(1, "D")


@dataclass
class ModuleBatch:
    module: str
    features: pd.DataFrame
    label: pd.Series
    label_kind: str                       # 'binary' | 'regression'
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    public_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    extra_labels: dict[str, pd.Series] = field(default_factory=dict)  # e.g. M3 overrun hours


def _rid(rng: np.random.Generator) -> str:
    return str(uuid.UUID(bytes=bytes(rng.integers(0, 256, size=16, dtype=np.uint8).tolist())))


def _dt_offsets(rng, n, lo_days, hi_days, base):
    offs = rng.integers(lo_days, hi_days, size=n)
    return base - offs * _DAY


# ===========================================================================
# M1 — Smart Quote Optimizer (reference; full raw tables)
# ===========================================================================
def simulate_m1(n: int, seed: int, tenant: str) -> ModuleBatch:
    rng = np.random.default_rng(seed)
    clock = get_clock()
    as_of = np.datetime64(clock.as_of().replace(tzinfo=None))

    cat_df, md_df, md = build_masterdata(tenant)
    wh_id = _rid(rng)
    warehouses = pd.DataFrame([{
        "id": wh_id, "warehouse_name": "Main", "short_name": "MN",
        "warehouse_type": "both", "status": "active",
    }])

    # products
    n_prod = max(12, n // 25)
    ptype_codes = ["FINISHED_GOOD", "SUB_ASSEMBLY", "RAW_MATERIAL", "SERVICE"]
    ptype_p = [0.5, 0.25, 0.15, 0.10]
    prod_rows = []
    for i in range(n_prod):
        sales = float(rng.uniform(100, 5000))
        margin0 = float(rng.uniform(0.18, 0.55))
        ptype = rng.choice(ptype_codes, p=ptype_p)
        prod_rows.append({
            "id": _rid(rng), "sku": f"P{i:04d}", "name": f"Product {i}",
            "product_type_id": md.id("PRODUCT_TYPE", ptype),
            "sales_price": round(sales, 2), "unit_cost": round(sales * (1 - margin0), 2),
            "project_type": "catalog_item", "route": "manufacture",
            "default_warehouse_id": wh_id, "on_hand": 0, "forecasted": 0, "reserved_quantity": 0,
        })
    products = pd.DataFrame(prod_rows)

    # salespeople (public.users) with latent skill tier; contacts with latent propensity
    n_sellers = max(4, n // 120)
    sellers = [{"id": _rid(rng), "email": f"seller{i}@demo.local", "name": f"Seller {i}",
                "auth_provider": "local", "status_id": md.id("STATUS", "ACTIVE")} for i in range(n_sellers)]
    seller_skill = {s["id"]: int(rng.integers(0, 5)) for s in sellers}

    n_contacts = max(10, n // 12)
    contacts = []
    contact_prop = {}
    for i in range(n_contacts):
        cid = _rid(rng)
        contacts.append({
            "id": cid, "name": f"Contact {i}", "email": f"contact{i}@demo.local",
            "life_cycle_status_id": md.id("CONTACT_LIFECYCLE_STATUS", "CUSTOMER"),
            "country_id": md.id("COUNTRY", "AU"),
        })
        contact_prop[cid] = float(np.clip(rng.beta(2, 2), 0.08, 0.92))
    contacts_df = pd.DataFrame(contacts)

    # quotations + line items
    q_rows, li_rows = [], []
    margin_frac, log_value, prop, days_left_frac, skill = [], [], [], [], []
    meta_created, meta_seller, meta_contact = [], [], []
    for _ in range(n):
        cid = rng.choice(list(contact_prop))
        sid = rng.choice([s["id"] for s in sellers])
        qid = _rid(rng)
        created = as_of - int(rng.integers(20, 400)) * _DAY
        valid_days = int(rng.integers(7, 45))
        expiration = (created + valid_days * _DAY).astype("datetime64[D]")
        nlines = int(rng.integers(1, 4))
        line_amt_total, qmargins, qweights = 0.0, [], []
        for k in range(nlines):
            p = products.iloc[int(rng.integers(0, len(products)))]
            qty = float(rng.integers(1, 20))
            negotiate = float(rng.uniform(0.80, 1.25))
            sales_price = round(float(p["sales_price"]) * negotiate, 4)
            line_amount = round(sales_price * qty, 2)
            line_amt_total += line_amount
            mf = (sales_price - float(p["unit_cost"])) / sales_price if sales_price > 0 else 0.0
            qmargins.append(mf)
            qweights.append(line_amount)
            li_rows.append({
                "id": _rid(rng), "quotation_id": qid, "product_id": p["id"],
                "quantity": qty, "unit_price": round(float(p["sales_price"]), 4),
                "sales_price": sales_price, "line_amount": line_amount, "sort_order": k,
            })
        wm = float(np.average(qmargins, weights=qweights)) if sum(qweights) > 0 else float(np.mean(qmargins))
        grand = round(line_amt_total * 1.10, 2)
        margin_frac.append(wm)
        log_value.append(np.log1p(grand))
        prop.append(contact_prop[cid])
        days_left_frac.append(valid_days / 45.0)
        skill.append(seller_skill[sid])
        meta_created.append(created)
        meta_seller.append(sid)
        meta_contact.append(cid)
        q_rows.append({"id": qid, "contact_id": cid, "sales_person_id": sid,
                       "created_at": created, "expiration_date": expiration, "grand_total": grand,
                       "total_amount": round(line_amt_total, 2)})

    # latent win draw
    z = GT.m1_win_logit(np.array(margin_frac), np.array(log_value), np.array(prop),
                        np.array(days_left_frac), np.array(skill), rng)
    p_win = GT.sigmoid(z)
    won_draw = rng.random(n) < p_win
    closed = rng.random(n) < 0.82  # the rest stay open (no label)

    stage_q = md.id("QUOTATION_STAGE", "QUOTATION")
    stage_sent = md.id("QUOTATION_STAGE", "QUOTATION_SENT")
    stage_so = md.id("QUOTATION_STAGE", "SALES_ORDER")
    st_draft = md.id("QUOTATION_STATUS", "DRAFT")
    st_sent = md.id("QUOTATION_STATUS", "QUOTATION_SENT")
    st_lost = md.id("QUOTATION_STATUS", "CLOSED_LOST")
    st_conf = md.id("QUOTATION_STATUS", "CONFIRMED")

    so_counter = 0
    for i, r in enumerate(q_rows):
        r["quotation_id"] = f"Q{i:05d}"
        r["quote_type"] = "Manual"
        r["payment_terms_id"] = md.id("PAYMENT_TERMS", "NET_30")
        r["organisation_id"] = None
        r["sales_order_id"] = None
        r["sent_at"] = None
        r["lost_at"] = None
        r["sales_order_created_at"] = None
        close_time = r["created_at"] + int(rng.integers(2, 25)) * _DAY
        if not closed[i]:
            r["stage_id"] = stage_sent if rng.random() < 0.5 else stage_q
            r["status_id"] = st_sent if r["stage_id"] == stage_sent else st_draft
        elif won_draw[i]:
            so_counter += 1
            r["stage_id"] = stage_so
            r["status_id"] = st_conf
            r["sales_order_id"] = f"S{so_counter:05d}"
            r["sales_order_created_at"] = close_time
        else:
            r["stage_id"] = stage_sent
            r["status_id"] = st_lost
            r["lost_at"] = close_time

    quotations = pd.DataFrame(q_rows)
    line_items = pd.DataFrame(li_rows)
    tables = {
        "master_data_category": cat_df, "master_data": md_df, "warehouses": warehouses,
        "products": products, "contacts": contacts_df,
        "quotations": quotations, "quotation_line_items": line_items,
    }

    # features via the REAL module feature builder (parity with train/score)
    from m1_quote.features import build_features, training_frame
    feats = build_features(tables, md, clock)
    X, y = training_frame(feats)
    return ModuleBatch(
        module="m1_quote", features=X, label=y, label_kind="binary",
        tables=tables, public_tables={"users": pd.DataFrame(sellers)},
        meta={"n_quotes": n, "n_closed": int(feats["is_closed"].sum()),
              "positive_rate": float(y.mean()), "md_map": md, "feature_frame": feats},
    )


# ===========================================================================
# M2/M3/M4 — feature+label contracts now; full raw tables with their modules
# ===========================================================================
def simulate_m2(n: int, seed: int, tenant: str) -> ModuleBatch:
    rng = np.random.default_rng(seed + 2)
    on_hand = rng.uniform(0, 400, n)
    rop = rng.uniform(20, 250, n)
    deficit = np.maximum(rop - on_hand, 0)
    consumption_rate = rng.gamma(2.0, 3.0, n)                # units/day (EWMA estimate)
    demand_var = rng.uniform(0.1, 1.2, n)                    # coefficient of variation
    incoming_cover = rng.uniform(0, 300, n) * (rng.random(n) < 0.6)  # open PO units
    vendor_reliability = np.clip(rng.beta(5, 2, n), 0, 1)    # historical on-time fraction
    days_cover = on_hand / np.maximum(consumption_rate, 1e-6)
    z = GT.m2_stockout_logit(deficit, consumption_rate, incoming_cover, vendor_reliability, demand_var, rng)
    y = (rng.random(n) < GT.sigmoid(z)).astype(int)
    X = pd.DataFrame({
        "consumption_rate": consumption_rate, "on_hand": on_hand, "rop": rop,
        "rop_deficit": deficit, "incoming_cover": incoming_cover,
        "vendor_reliability": vendor_reliability, "demand_var": demand_var, "days_cover": days_cover,
    })
    return ModuleBatch(module="m2_inventory", features=X, label=pd.Series(y, name="stockout_30d"),
                       label_kind="binary",
                       meta={"positive_rate": float(y.mean()),
                             "note": "raw PO/GRN/consumption tables built in checkpoint 3"})


def simulate_m3(n: int, seed: int, tenant: str) -> ModuleBatch:
    rng = np.random.default_rng(seed + 3)
    pace_ratio = rng.lognormal(0.0, 0.35, n)         # actual-min@25% / (expected*0.25)
    load_ratio = rng.gamma(2.0, 0.5, n)              # concurrency / capacity
    skill_tier = rng.integers(0, 5, n).astype(float)
    material_shortfall = np.clip(rng.normal(0.2, 0.25, n), 0, 1)
    complexity = rng.gamma(3.0, 1.5, n)              # ops + components + dep depth
    z = GT.m3_delay_logit(pace_ratio, load_ratio, skill_tier, material_shortfall, complexity, rng)
    p = GT.sigmoid(z)
    y = (rng.random(n) < p).astype(int)
    # overrun hours (regressor head): positive only when delayed-ish, noisy
    base_expected_h = rng.uniform(4, 40, n)
    overrun = np.maximum(0.0, (pace_ratio - 1.0) * base_expected_h + rng.normal(0, 2, n)) * (p > 0.4)
    X = pd.DataFrame({
        "pace_ratio": pace_ratio, "load_ratio": load_ratio, "skill_tier": skill_tier,
        "material_shortfall": material_shortfall, "complexity": complexity,
        "expected_hours": base_expected_h,
    })
    return ModuleBatch(module="m3_delay", features=X, label=pd.Series(y, name="delay"),
                       label_kind="binary", extra_labels={"overrun_hours": pd.Series(overrun, name="overrun_hours")},
                       meta={"positive_rate": float(y.mean()),
                             "note": "raw WO/time-log tables built in checkpoint 3"})


def simulate_m4(n: int, seed: int, tenant: str) -> ModuleBatch:
    rng = np.random.default_rng(seed + 4)
    # error types mixed so NO single feature explains the whole label (anti-leak).
    err_type = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.78, 0.06, 0.06, 0.05, 0.05])
    # 0=clean, 1=duplicate, 2=uom_mismatch, 3=qty_outlier, 4=name_typo
    qty_z = np.abs(rng.normal(0, 1, n))
    qty_z[err_type == 3] += rng.uniform(3, 6, (err_type == 3).sum())   # outliers
    uom_mismatch = (err_type == 2).astype(float)
    # a few false-positive uom flags + missed ones (noise) so it isn't deterministic
    flip = rng.random(n) < 0.05
    uom_mismatch = np.where(flip, 1 - uom_mismatch, uom_mismatch)
    duplicate = (err_type == 1).astype(float)
    duplicate = np.where(rng.random(n) < 0.05, 1 - duplicate, duplicate)
    name_sim = rng.uniform(0.7, 1.0, n)
    name_sim[err_type == 4] = rng.uniform(0.2, 0.6, (err_type == 4).sum())  # typo -> low similarity
    freq_normalcy = np.clip(rng.normal(0.7, 0.2, n), 0, 1)
    freq_normalcy[err_type != 0] -= rng.uniform(0, 0.3, (err_type != 0).sum())
    y = (err_type != 0).astype(int)
    X = pd.DataFrame({
        "quantity_zscore": qty_z, "uom_mismatch": uom_mismatch, "duplicate": duplicate,
        "name_similarity": name_sim, "freq_normalcy": freq_normalcy,
    })
    return ModuleBatch(module="m4_bom", features=X, label=pd.Series(y, name="is_error"),
                       label_kind="binary",
                       meta={"positive_rate": float(y.mean()), "err_type": err_type,
                             "note": "raw BOM/component tables + TF-IDF/RapidFuzz fe built in checkpoint 3"})


# Built-in feature/label proxies (used until a module ships its own raw-table
# builder). A module gains a full raw-table generator by providing
# ``modules/<module>/synth.py`` with ``build(n, seed, tenant) -> ModuleBatch`` —
# dispatch prefers it (so M2/M3/M4 light up automatically as they land), with NO
# edit to this shared file (keeps the fan-out conflict-free).
_PROXIES = {
    "m1_quote": simulate_m1,
    "m2_inventory": simulate_m2,
    "m3_delay": simulate_m3,
    "m4_bom": simulate_m4,
}
_DEFAULT_N = {"m1_quote": 1600, "m2_inventory": 1500, "m3_delay": 1500, "m4_bom": 2200}


def _module_builder(module: str):
    import importlib
    try:
        mod = importlib.import_module(f"{module}.synth")
    except Exception:
        return None
    return getattr(mod, "build", None)


def simulate(module: str, n: int | None = None, seed: int = 7, tenant: str = "demo") -> ModuleBatch:
    if module not in _PROXIES:
        raise KeyError(f"unknown module {module!r}")
    n = n or _DEFAULT_N[module]
    builder = _module_builder(module)
    if builder is not None:
        return builder(n, seed, tenant)
    return _PROXIES[module](n, seed, tenant)
