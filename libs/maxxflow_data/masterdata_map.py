"""Load the ``(category_code, code) -> uuid`` map from the tenant's MasterData.

Mirrors the synth seeder's map but sourced from Postgres, so feature/label code
resolves labels by code (never hard-coded UUIDs). Duck-compatible with the synth
``MasterDataMap`` (both expose ``.id(category, code)``)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MasterDataMap:
    code_to_id: dict[tuple[str, str], str] = field(default_factory=dict)

    def id(self, category: str, code: str) -> str:
        return self.code_to_id[(category, code)]


def load_md_map(da, tenant: str) -> MasterDataMap:
    sql = (
        "SELECT c.code AS category_code, m.code AS code, m.id AS id "
        "FROM master_data m JOIN master_data_category c ON m.category_id = c.id "
        "WHERE m.deleted_at IS NULL AND m.code IS NOT NULL"
    )
    df = da.query(sql, tenant=tenant)
    m = MasterDataMap()
    for _, r in df.iterrows():
        m.code_to_id[(r["category_code"], r["code"])] = str(r["id"])
    return m
