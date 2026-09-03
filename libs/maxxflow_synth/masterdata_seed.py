"""Seed MasterData first (plan §1a row 2): labels are defined by WHICH UUID is
written, so the generator builds ``MasterDataCategory`` + ``MasterData`` and holds
a ``(category, code) -> uuid`` map. UUIDs are deterministic (uuid5) so the same
code maps to the same id across runs and environments (reproducible labels)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import pandas as pd

from maxxflow_core import masterdata as MD

_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # fixed namespace


def _u(*parts: str) -> str:
    return str(uuid.uuid5(_NS, ":".join(parts)))


@dataclass
class MasterDataMap:
    tenant: str
    code_to_id: dict[tuple[str, str], str] = field(default_factory=dict)

    def id(self, category: str, code: str) -> str:
        return self.code_to_id[(category, code)]


def build_masterdata(tenant: str) -> tuple[pd.DataFrame, pd.DataFrame, MasterDataMap]:
    cat_rows, md_rows = [], []
    mapping = MasterDataMap(tenant=tenant)
    for cat_code, cat in MD.CATEGORIES.items():
        cat_id = _u(tenant, "cat", cat_code)
        cat_rows.append({
            "id": cat_id, "name": cat["name"], "description": None, "code": cat_code,
            "status": "active", "is_system_fixed": True,
        })
        for code, code_name in cat["codes"].items():
            md_id = _u(tenant, cat_code, code)
            mapping.code_to_id[(cat_code, code)] = md_id
            md_rows.append({
                "id": md_id, "category_id": cat_id, "name": code_name, "description": None,
                "code": code, "status": "active", "parent_id": None, "metadata": None,
                "is_system_fixed": True,
            })
    return pd.DataFrame(cat_rows), pd.DataFrame(md_rows), mapping
