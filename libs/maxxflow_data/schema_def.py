"""Tenant-template schema as data — the single source of truth (plan §1, §3).

Hand-translated from ``schema.prisma`` for the tables the four AI/ML modules read
and write. ONE spec drives three things, so they can never drift apart:

* ``tenant_template_ddl(schema)``         -> CREATE TABLE SQL (the local provisioner)
* the pandera schemas in gate A           -> type / Decimal-scale / nullability checks
* the generator↔DAL column contract test  -> generator output cols == DAL SELECT cols

Fidelity focus (plan §1a): exact ``Decimal(p,s)`` scales, ``String[]`` arrays,
``JsonB`` columns, the STALE ``items.rop_status`` column (present so the
"never read" guard can prove it is never selected), and ``deleted_at`` soft-delete
columns. Bookkeeping columns (created_at/updated_at/created_by) get DB defaults so
the generator need not supply them.

NOTE (scoping): purely-integration columns (integration_*, stripe_*, superset_*)
and tables unrelated to M1–M4 are intentionally omitted from the LOCAL DDL. The
real app provisions the full schema via Prisma migrate; this template is the
AI/ML-relevant subset. Documented in NOTES.md.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Col:
    name: str
    type: str          # logical type -> mapped to Postgres + pandas
    nullable: bool = True
    default: str | None = None  # raw SQL default


def C(name, type, nullable=True, default=None):  # terse constructor
    return Col(name, type, nullable, default)


_ID = C("id", "uuid", nullable=False, default="gen_random_uuid()")
_CREATED = C("created_at", "timestamp", nullable=False, default="now()")
_UPDATED = C("updated_at", "timestamp", nullable=False, default="now()")
_CREATED_BY = C("created_by", "uuid", nullable=False, default="'00000000-0000-0000-0000-000000000000'")
_DELETED = C("deleted_at", "timestamp", nullable=True)
_CUSTOM = C("custom_elements", "jsonb", nullable=True)

# table -> ordered columns. Tenant schema unless prefixed in PUBLIC_TABLES.
TENANT_TABLES: dict[str, list[Col]] = {
    "master_data_category": [
        _ID, C("name", "varchar(255)", False), C("description", "varchar(255)"),
        C("code", "varchar(255)"), C("status", "varchar(10)", False), C("is_system_fixed", "bool", False, "false"),
        _CREATED, _UPDATED, _DELETED,
    ],
    "master_data": [
        _ID, C("category_id", "uuid", False), C("name", "varchar(255)", False),
        C("description", "varchar(255)"), C("code", "varchar(255)"), C("status", "varchar(10)", False),
        C("parent_id", "uuid"), C("metadata", "jsonb"), C("is_system_fixed", "bool", False, "false"),
        _CREATED, _UPDATED, _DELETED,
    ],
    "warehouses": [
        _ID, C("warehouse_name", "varchar(255)", False), C("short_name", "varchar(50)", False),
        C("warehouse_type", "varchar(20)", False, "'both'"), C("status", "varchar(20)", False, "'active'"),
        _CUSTOM, _CREATED, _UPDATED, _CREATED_BY, _DELETED,
    ],
    "products": [
        _ID, C("sku", "varchar(100)", False), C("name", "varchar(255)", False),
        C("product_type_id", "uuid", False), C("sales_price", "decimal(12,2)", False),
        C("unit_cost", "decimal(12,2)", False), C("project_type", "varchar(50)", False, "'catalog_item'"),
        C("route", "varchar(50)", False, "'manufacture'"), C("default_warehouse_id", "uuid", False),
        C("on_hand", "decimal(12,2)", False, "0"), C("forecasted", "decimal(12,2)", False, "0"),
        C("reserved_quantity", "decimal(12,2)", False, "0"), C("notes", "text"),
        _CUSTOM, _CREATED, _UPDATED, _CREATED_BY, _DELETED,
    ],
    "items": [
        _ID, C("item_name", "varchar(255)", False), C("part_number", "varchar(100)"),
        C("item_type", "varchar(50)", False, "'Goods'"), C("unit_cost", "decimal(18,4)"),
        C("unit_of_measurement", "varchar(50)"), C("rop", "int", True, "0"),
        C("default_warehouse_id", "uuid"), C("available_quantity", "decimal(10,2)", False, "0"),
        C("forecasted_quantity", "decimal(10,2)", False, "0"),
        # STALE per schema.prisma — present so the "never read rop_status" guard is real.
        C("rop_status", "varchar(20)", False, "'Available'"),
        C("status", "varchar(20)", False, "'Active'"),
        _CUSTOM, _CREATED, _UPDATED, _CREATED_BY, _DELETED,
    ],
    "item_vendors": [
        _ID, C("item_id", "uuid", False), C("vendor_id", "uuid", False),
        C("vendor_name", "varchar(255)"), C("lead_time_days", "int"), C("unit_price", "decimal(18,4)"),
        _CREATED, _UPDATED,
    ],
    "contacts": [
        _ID, C("name", "varchar(255)", False), C("email", "varchar(255)", False),
        C("life_cycle_status_id", "uuid", False), C("country_id", "uuid"),
        _CUSTOM, C("added_date", "timestamp", False, "now()"), _CREATED, _UPDATED, _CREATED_BY, _DELETED,
    ],
    "organisations": [
        _ID, C("company_name", "varchar(255)", False), C("company_type_id", "uuid", False),
        C("email", "varchar(255)", False), C("country_id", "uuid", False), C("status_id", "uuid", False),
        _CUSTOM, _CREATED, _UPDATED, _CREATED_BY, _DELETED,
    ],
    "quotations": [
        _ID, C("quotation_id", "varchar(20)", False), C("sales_order_id", "varchar(20)"),
        C("contact_id", "uuid"), C("organisation_id", "uuid"),
        C("stage_id", "uuid", False), C("status_id", "uuid", False),
        C("quote_type", "varchar(30)", False, "'Manual'"), C("expiration_date", "date"),
        C("payment_terms_id", "uuid"), C("total_amount", "decimal(18,2)", False, "0"),
        C("tax_percentage", "decimal(5,2)"), C("tax_amount", "decimal(18,2)"),
        C("grand_total", "decimal(18,2)", False, "0"), C("sales_person_id", "uuid", False),
        C("sent_at", "timestamp"), C("lost_at", "timestamp"), C("sales_order_created_at", "timestamp"),
        _CUSTOM, _CREATED, _UPDATED, _CREATED_BY, _DELETED,
    ],
    "quotation_line_items": [
        _ID, C("quotation_id", "uuid", False), C("product_id", "uuid", False),
        C("quantity", "decimal(18,4)", False), C("unit_price", "decimal(18,4)", False),
        C("sales_price", "decimal(18,4)", False), C("line_amount", "decimal(18,2)", False),
        C("sort_order", "int", False, "0"), _CREATED, _UPDATED,
    ],
    "boms": [
        _ID, C("code", "varchar(20)", False), C("name", "varchar(255)", False),
        C("product_id", "uuid"), C("description", "text"), C("version", "varchar(50)"),
        C("locked", "bool", False, "false"), _CUSTOM, _CREATED, _UPDATED, _CREATED_BY, _DELETED,
    ],
    "bom_components": [
        _ID, C("bom_id", "uuid", False), C("item_id", "uuid"), C("product_id", "uuid"),
        C("component_type", "varchar(10)", False, "'item'"), C("quantity", "decimal(12,4)", False),
        C("unit_of_measure", "varchar(50)"), C("notes", "text"), _CREATED, _UPDATED, _CREATED_BY,
    ],
    "work_centers": [
        _ID, C("name", "varchar(255)", False), C("code", "varchar(50)", False),
        C("working_hours_id", "uuid", False), C("setup_time", "float"), C("cleanup_time", "float"),
        C("cost_per_hour", "float"), C("employee_cost_per_hour", "float"),
        C("allowed_employees", "text[]", False, "'{}'"), C("description", "text"),
        _CUSTOM, _CREATED, _UPDATED, _CREATED_BY, _DELETED,
    ],
    "operations": [
        # operationName/workCenterId/estimatedDuration have NO @map in schema.prisma,
        # so Prisma provisions them as case-preserving camelCase columns. We mirror
        # that here (DDL quotes identifiers); the M3 DAL aliases them to snake_case.
        _ID, C("bom_id", "uuid", False), C("operationName", "varchar(255)", False),
        C("workCenterId", "uuid", False), C("estimatedDuration", "int", False, "0"),
        C("operation_type_id", "uuid", False), C("allowed_employees", "uuid[]", False, "'{}'"),
        _CUSTOM, _CREATED, _UPDATED, _DELETED,
    ],
    "operation_dependencies": [
        _ID, C("operation_id", "uuid", False), C("depends_on_id", "uuid", False), _CREATED,
    ],
    "manufacturing_orders": [
        _ID, C("reference", "varchar(20)", False), C("product_id", "uuid", False),
        C("bom_id", "uuid"), C("quantity", "decimal(12,4)", False), C("scheduled_date", "timestamp"),
        C("status_id", "uuid", False), C("component_status_id", "uuid", False),
        C("warehouse_id", "uuid"), C("quotation_id", "uuid"),
        C("confirmed_at", "timestamp"), C("completed_at", "timestamp"), C("cancelled_at", "timestamp"),
        _CUSTOM, _CREATED, _UPDATED, _CREATED_BY, _DELETED,
    ],
    "mo_components": [
        _ID, C("mo_id", "uuid", False), C("item_id", "uuid"), C("product_id", "uuid"),
        C("component_type", "varchar(10)", False, "'item'"), C("required_qty", "decimal(12,4)", False),
        C("reserved_qty", "decimal(12,4)", False, "0"), C("consumed_qty", "decimal(12,4)", False, "0"),
        C("availability_id", "uuid", False), C("purchase_order_id", "uuid"), _CREATED, _UPDATED,
    ],
    "work_orders": [
        _ID, C("mo_id", "uuid", False), C("operation_id", "uuid", False),
        C("work_center_id", "uuid", False), C("quantity", "decimal(12,4)", False),
        C("units_done", "decimal(12,4)", False, "0"), C("expected_duration", "int", False),
        C("real_duration", "int"), C("scheduled_start", "timestamp"), C("scheduled_end", "timestamp"),
        C("actual_start", "timestamp"), C("actual_end", "timestamp"),
        C("assigned_operators", "text[]", False, "'{}'"), C("status_id", "uuid", False),
        _CREATED, _UPDATED, _CREATED_BY,
    ],
    "work_order_time_logs": [
        _ID, C("work_order_id", "uuid", False), C("operator_id", "uuid", False),
        C("started_at", "timestamp", False), C("ended_at", "timestamp"),
        C("duration_minutes", "int"), _CREATED,
    ],
    "purchase_orders": [
        _ID, C("reference_no", "varchar(50)", False), C("vendor_id", "uuid"), C("vendor_name", "varchar(255)"),
        C("organisation_id", "uuid"), C("warehouse_id", "uuid"),
        C("scheduled_delivery_date", "timestamp"), C("status", "varchar(50)", False, "'Draft'"),
        _CUSTOM, C("sent_at", "timestamp"), _CREATED, _UPDATED,
    ],
    "purchase_order_lines": [
        _ID, C("purchase_order_id", "uuid", False), C("item_id", "uuid", False),
        C("item_name", "varchar(255)"), C("ordered_quantity", "decimal(10,2)", False),
        C("received_quantity", "decimal(10,2)", False, "0"), C("unit_price", "decimal(18,4)", False),
        _CREATED, _UPDATED,
    ],
    "goods_received_notes": [
        # bill_created / bill_created_at are on purchase_orders in schema.prisma, NOT
        # the GRN — so the GRN status-transition timestamp is updated_at.
        _ID, C("reference_no", "varchar(50)", False), C("purchase_order_id", "uuid", False),
        C("vendor_id", "uuid", False), C("warehouse_id", "uuid", False),
        C("scheduled_delivery_date", "timestamp"), C("status", "varchar(50)", False, "'Draft'"),
        C("is_partially_closed", "bool", False, "false"), _CREATED, _UPDATED,
    ],
    "grn_lines": [
        _ID, C("grn_id", "uuid", False), C("purchase_order_line_id", "uuid", False),
        C("item_id", "uuid", False), C("ordered_quantity", "decimal(10,2)", False),
        C("received_quantity", "decimal(10,2)", False, "0"), C("unit_price", "decimal(18,4)", False),
        _CREATED, _UPDATED,
    ],
    "audit_logs": [
        _ID, C("user_id", "uuid"), C("module", "varchar(100)", False), C("action", "varchar(50)", False),
        C("entity_type", "varchar(100)"), C("entity_id", "uuid"), C("metadata", "jsonb"),
        C("timestamp", "timestamp", False, "now()"), _CREATED,
    ],
}

# Minimal public schema for cross-schema operator/salesperson refs (PII source).
PUBLIC_TABLES: dict[str, list[Col]] = {
    "users": [
        _ID, C("email", "varchar(255)", False), C("name", "varchar(255)"),
        C("auth_provider", "varchar(20)", False, "'local'"), C("status_id", "uuid", False),
        _CREATED, _UPDATED, _DELETED,
    ],
    "tenants": [
        _ID, C("slug", "varchar(100)", False), C("name", "varchar(255)", False),
        C("schema_name", "varchar(63)", False), C("status_id", "uuid", False),
        _CREATED, _UPDATED, _DELETED,
    ],
}


def columns(table: str) -> list[str]:
    spec = TENANT_TABLES.get(table) or PUBLIC_TABLES.get(table)
    if spec is None:
        raise KeyError(table)
    return [c.name for c in spec]


def all_tenant_tables() -> list[str]:
    return list(TENANT_TABLES.keys())
