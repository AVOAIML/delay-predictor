"""Canonical MasterData categories + codes (plan §1a row 2).

There are NO business enums in the schema: every status/stage/type is a row in
``MasterData`` (``category`` + ``code``), referenced by FK. Therefore a *label*
is defined by WHICH MasterData UUID is written. The synthetic generator's first
job per tenant is to seed ``MasterDataCategory`` + ``MasterData`` from this
registry and hold a ``(category, code) -> uuid`` map; all feature/label SQL
resolves codes through that map — never hard-coded UUIDs, never raw strings
scattered through the modules.
"""

from __future__ import annotations

# category_code -> {display name, {code: display name}}
CATEGORIES: dict[str, dict] = {
    "STATUS": {"name": "Status", "codes": {"ACTIVE": "Active", "INACTIVE": "Inactive"}},
    "COUNTRY": {"name": "Country", "codes": {"AU": "Australia", "NZ": "New Zealand", "US": "United States"}},
    "PRODUCT_TYPE": {
        "name": "Product Type",
        "codes": {
            "FINISHED_GOOD": "Finished Good",
            "SUB_ASSEMBLY": "Sub Assembly",
            "RAW_MATERIAL": "Raw Material",
            "SERVICE": "Service",
        },
    },
    "PAYMENT_TERMS": {
        "name": "Payment Terms",
        "codes": {"NET_30": "Net 30", "NET_60": "Net 60", "COD": "Cash on Delivery"},
    },
    # M1 — Quote. Dual status/stage system (see schema comment on Quotation).
    "QUOTATION_STAGE": {
        "name": "Quotation Stage",
        "codes": {"QUOTATION": "Quotation", "QUOTATION_SENT": "Quotation Sent", "SALES_ORDER": "Sales Order"},
    },
    "QUOTATION_STATUS": {
        "name": "Quotation Status",
        "codes": {
            "DRAFT": "Draft",
            "QUOTATION_SENT": "Quotation Sent",
            "CLOSED_LOST": "Closed Lost",
            "CONFIRMED": "Confirmed",  # won (sales order created)
        },
    },
    "ORGANISATION_TYPE": {
        "name": "Organisation Type",
        "codes": {"VENDOR": "Vendor", "CUSTOMER": "Customer", "BOTH": "Both"},
    },
    "CONTACT_LIFECYCLE_STATUS": {
        "name": "Contact Lifecycle Status",
        "codes": {"CUSTOMER": "Customer", "CONTACT": "Contact"},
    },
    # M3 — Delay.
    "MO_STATUS": {
        "name": "Manufacturing Order Status",
        "codes": {
            "DRAFT": "Draft",
            "CONFIRMED": "Confirmed",
            "IN_PROGRESS": "In Progress",
            "DONE": "Done",
            "CANCELLED": "Cancelled",
        },
    },
    "MO_COMPONENT_STATUS": {
        "name": "MO Component Status",
        "codes": {
            "AVAILABLE": "Available",
            "PARTIALLY_AVAILABLE": "Partially Available",
            "NOT_AVAILABLE": "Not Available",
        },
    },
    "WORK_ORDER_STATUS": {
        "name": "Work Order Status",
        "codes": {
            "PENDING": "Pending",
            "READY": "Ready",
            "IN_PROGRESS": "In Progress",
            "DONE": "Done",
            "CANCELLED": "Cancelled",
        },
    },
    "WORKING_HOURS": {
        "name": "Working Hours",
        "codes": {"STANDARD_8H": "Standard 8h", "SHIFT_16H": "Two Shift 16h", "CONTINUOUS_24H": "Continuous 24h"},
    },
    "OPERATION_TYPE": {
        "name": "Operation Type",
        "codes": {"INDEPENDENT": "Independent", "DEPENDENT": "Dependent"},
    },
}

# --- semantic label codes referenced by feature/label/score code -------------
# M1: Win = sales order created (stage SALES_ORDER); Loss = CLOSED_LOST.
QUOTATION_WIN_STAGE = "SALES_ORDER"
QUOTATION_WIN_STATUS = "CONFIRMED"
QUOTATION_LOSS_STATUS = "CLOSED_LOST"
# The live MRP API uses the *_STATUS codes, while older/synthetic datasets use
# the shorter codes above. Accept both during the migration so either tenant
# representation can be labelled correctly.
QUOTATION_WIN_STATUS_CODES = ("SALES_ORDER_STATUS", QUOTATION_WIN_STATUS)
QUOTATION_LOSS_STATUS_CODES = ("CLOSED_LOST_STATUS", QUOTATION_LOSS_STATUS)
# M3: label measured at MO completion; trigger fires on MO -> DONE.
MO_DONE = "DONE"
WORK_ORDER_DONE = "DONE"
WORK_ORDER_IN_PROGRESS = "IN_PROGRESS"


def resolve_ids(md, category: str, codes: tuple[str, ...]) -> set[str]:
    """Resolve every available MasterData ID among compatible code aliases."""
    resolved: set[str] = set()
    for code in codes:
        try:
            resolved.add(md.id(category, code))
        except KeyError:
            continue
    if not resolved:
        raise KeyError((category, codes))
    return resolved


def iter_entries():
    """Yield ``(category_code, category_name, code, code_name)`` for every entry."""
    for cat_code, cat in CATEGORIES.items():
        for code, code_name in cat["codes"].items():
            yield cat_code, cat["name"], code, code_name


def category_codes(category_code: str) -> list[str]:
    return list(CATEGORIES[category_code]["codes"].keys())
