"""M3 — Production Delay (data-access + rollup + elements layer only so far).

``fe`` / ``train`` / ``score`` / ``drift`` are not implemented yet — this
module currently exposes ``read_delay_tables`` (the raw-table read described
in ``dal.py``'s docstring: available / partial / not-available columns),
``build_job_rollups`` (joins those raw tables into the job/operations/
operators/components rollup shape, described in ``rollup.py``'s docstring),
and ``calculate_delay_elements`` (the element_1/2/3/7/9 risk signals computed
from that rollup shape, described in ``elements.py``'s docstring).
"""

from m3_production_delay.rule_engine.dal import read_delay_tables
from m3_production_delay.rule_engine.rollup import build_job_rollups
from m3_production_delay.rule_engine.elements import (
    calculate_delay_elements_for_jobs,
    composite_risk_score,
    is_delayed,
    manufacturing_order_progress,
    DEFAULT_RISK_WEIGHTS,
    DEFAULT_DELAY_THRESHOLD,
)

__all__ = [
    "read_delay_tables", "build_job_rollups",
    "calculate_delay_elements_for_jobs",
    "composite_risk_score", "is_delayed", "manufacturing_order_progress",
    "DEFAULT_RISK_WEIGHTS", "DEFAULT_DELAY_THRESHOLD",
]
