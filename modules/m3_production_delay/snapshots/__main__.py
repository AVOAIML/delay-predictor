"""``python -m m3_production_delay.snapshots`` -> the outcome sweep CLI.

Snapshots need no command: ``review.pipeline.run`` writes them on every
review. Outcomes do, because an MO completes long after it was scored — this is
the entry point a scheduler (cron, a Container Apps job) runs periodically.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from dataclasses import asdict

from m3_production_delay.snapshots.outcomes import DEFAULT_LOOKBACK, sweep_outcomes


def _run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m m3_production_delay.snapshots",
        description=(
            "Record the outcome of every manufacturing order completed since the last sweep, "
            "for one tenant, into the lake."
        ),
    )
    parser.add_argument("--tenant", default="demo")
    parser.add_argument(
        "--lookback-hours",
        type=float,
        default=DEFAULT_LOOKBACK.total_seconds() / 3600,
        help="how far before the bookmark to re-check, to catch replica lag (default: 24)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and log every outcome, write nothing, leave the bookmark where it is",
    )
    args = parser.parse_args(argv)
    report = sweep_outcomes(
        args.tenant,
        lookback=_dt.timedelta(hours=args.lookback_hours),
        dry_run=args.dry_run,
    )
    print(json.dumps(asdict(report), indent=2))
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_cli())
