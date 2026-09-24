"""M3 training-data collection: snapshots at scoring time, outcomes at completion.

Two record types, both written to the tenant's own prefix in the lake (bronze):

* **snapshot** (``snapshot.py``) — one per scoring event. Every signal value,
  the weights and threshold used, and the full validated insight. Written by
  ``review.pipeline.run`` right after the advisory writeback.
* **outcome** (``outcomes.py``) — one per completed MO. When it finished
  against its planned finish, and the raw timings of every work order.
  Written by a watermarked sweep, ``python -m m3_production_delay.snapshots``.

Joined on ``job_id``, they give "these signals at this point in the MO's
progress, then it finished this late" — the history the Weight Agent's
fitted-weights path reports as missing today
(``history_admissibility=inadmissible``). Nothing here trains anything; it
collects what training will need.
"""

from m3_production_delay.snapshots.outcomes import SweepReport, label_outcome, sweep_outcomes
from m3_production_delay.snapshots.snapshot import SnapshotReport, build_snapshot, write_snapshots
from m3_production_delay.snapshots.store import job_key, outcome_name, snapshot_name

__all__ = [
    "SnapshotReport",
    "SweepReport",
    "build_snapshot",
    "job_key",
    "label_outcome",
    "outcome_name",
    "snapshot_name",
    "sweep_outcomes",
    "write_snapshots",
]
