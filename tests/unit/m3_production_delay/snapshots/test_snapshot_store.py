"""Snapshot layout and strict-JSON conversion."""

from __future__ import annotations

import datetime as dt
import decimal
import json
import math
import re
import uuid

import numpy as np
import pandas as pd
import pytest

from m3_production_delay.snapshots.store import (
    POSITIVE_INFINITY,
    job_key,
    outcome_name,
    snapshot_name,
    to_jsonable,
)

AEST = dt.timezone(dt.timedelta(hours=10))


def test_a_job_key_is_filename_safe_and_readable():
    key = job_key("WH/MO/00142")
    assert re.fullmatch(r"[A-Za-z0-9._-]+", key)
    assert key.startswith("WH_MO_00142-")


def test_a_job_key_is_deterministic():
    assert job_key("WH/MO/00142") == job_key("WH/MO/00142")


def test_references_that_sanitize_alike_never_share_a_key():
    # Both read as "WH_MO_1" once the slash is replaced.
    assert job_key("WH/MO/1") != job_key("WH_MO_1")


def test_a_snapshot_is_partitioned_by_the_utc_date_of_scoring():
    # 02:15 on the 24th in +10:00 is still the 23rd in UTC.
    name = snapshot_name("WH/MO/00142", dt.datetime(2026, 9, 24, 2, 15, tzinfo=AEST))
    assert name.startswith("snapshots/dt=2026-09-23/WH_MO_00142-")


def test_two_scorings_of_one_mo_never_share_a_name():
    first = snapshot_name("WH/MO/00142", dt.datetime(2026, 9, 24, 9, 0, tzinfo=AEST))
    later = snapshot_name("WH/MO/00142", dt.datetime(2026, 9, 24, 13, 0, tzinfo=AEST))
    assert first != later


def test_a_naive_scoring_instant_is_refused():
    with pytest.raises(ValueError):
        snapshot_name("WH/MO/00142", dt.datetime(2026, 9, 24, 9, 0))


def test_one_mo_has_exactly_one_outcome_name():
    assert outcome_name("WH/MO/00142") == outcome_name("WH/MO/00142")
    assert outcome_name("WH/MO/00142").startswith("outcomes/WH_MO_00142-")


def test_engine_values_become_strict_json():
    op_id = uuid.uuid4()
    converted = to_jsonable(
        {
            "np_float": np.float64(1.5),
            "np_int": np.int64(3),
            "np_bool": np.bool_(True),
            "decimal": decimal.Decimal("8.0000"),
            "uuid": op_id,
            "timestamp": pd.Timestamp("2026-09-24T09:00:00"),
            "nat": pd.NaT,
            "tuple": (1, 2),
            op_id: "uuid key",
        }
    )
    assert converted == {
        "np_float": 1.5,
        "np_int": 3,
        "np_bool": True,
        "decimal": 8.0,
        "uuid": str(op_id),
        "timestamp": "2026-09-24T09:00:00",
        "nat": None,
        "tuple": [1, 2],
        str(op_id): "uuid key",
    }
    json.dumps(converted, allow_nan=False)


def test_an_infinite_ratio_is_kept_readable_and_nan_is_no_value():
    # A zero-stock component makes material_shortfall_ratio infinite: a real
    # value the score depends on, so it must survive — and float() reads it back.
    converted = to_jsonable({"shortfall": math.inf, "down": -math.inf, "missing": math.nan})
    assert converted == {"shortfall": POSITIVE_INFINITY, "down": "-inf", "missing": None}
    assert math.isinf(float(converted["shortfall"]))


def test_an_unknown_type_is_an_error_not_a_silent_str():
    with pytest.raises(TypeError):
        to_jsonable({"x": object()})
