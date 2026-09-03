"""JSON serialization helper. Model outputs come back through pandas as NumPy
scalar types (``np.bool_`` / ``np.integer`` / ``np.floating``) which ``json.dumps``
can't serialize; use ``json.dumps(obj, default=json_default)`` for all writebacks."""

from __future__ import annotations

import numpy as np


def json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)!r}")
