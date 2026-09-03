"""Single as-of clock (plan §1a timestamps row, §12a #7).

Schema timestamps are ``Timestamp(6)`` = TZ-naive. If date math ever calls
``datetime.now()`` / ``date.today()`` without an explicit timezone, the worker's
``TZ`` env leaks into days-to-expiration (M1) and the 25%-milestone clock (M3).

Rules enforced here:

* Every "current time" comes from a :class:`Clock`, never bare ``now()``.
* Naive DB timestamps are interpreted in a FIXED ``presentation_tz`` — never the
  ambient process tz — so ``TZ=UTC`` and ``TZ=Australia/Sydney`` yield identical
  features. ``tests/unit/test_clock_tz_invariance.py`` asserts exactly this.
* ``as_of`` can be frozen via ``AS_OF_OVERRIDE`` for reproducible runs/tests.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from dateutil import parser as _dateparser


@dataclass(frozen=True)
class Clock:
    """Deterministic, tz-explicit clock. Build via :func:`get_clock`."""

    presentation_tz: str = "Australia/Sydney"
    as_of_override: str = ""  # ISO-8601; empty => real now()

    @property
    def _tz(self) -> ZoneInfo:
        return ZoneInfo(self.presentation_tz)

    def now_utc(self) -> _dt.datetime:
        """Real wall clock as tz-aware UTC. The ONLY place we read the OS clock."""
        return _dt.datetime.now(tz=_dt.timezone.utc)

    def as_of(self) -> _dt.datetime:
        """The as-of instant for this run, tz-aware in ``presentation_tz``.

        Override-driven when ``AS_OF_OVERRIDE`` is set (reproducible), else now().
        """
        if self.as_of_override:
            parsed = _dateparser.isoparse(self.as_of_override)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=self._tz)
            return parsed.astimezone(self._tz)
        return self.now_utc().astimezone(self._tz)

    def as_of_date(self) -> _dt.date:
        return self.as_of().date()

    def localize(self, naive: _dt.datetime | None) -> _dt.datetime | None:
        """Interpret a naive DB timestamp in the fixed presentation tz (tz-aware)."""
        if naive is None:
            return None
        if naive.tzinfo is not None:
            return naive.astimezone(self._tz)
        return naive.replace(tzinfo=self._tz)

    def days_between(self, start: _dt.datetime | _dt.date | None,
                     end: _dt.datetime | _dt.date | None) -> float | None:
        """Signed fractional days ``end - start``. tz-explicit, ambient-tz-free."""
        if start is None or end is None:
            return None
        s = self._to_aware(start)
        e = self._to_aware(end)
        return (e - s).total_seconds() / 86400.0

    def days_to_expiration(self, expiration: _dt.datetime | _dt.date | None,
                           created: _dt.datetime | _dt.date | None) -> float | None:
        """M1 days-to-expiration = expiration - created (quote validity window)."""
        return self.days_between(created, expiration)

    def days_until(self, target: _dt.datetime | _dt.date | None) -> float | None:
        """Signed days from ``as_of`` to ``target`` (e.g. PO scheduled delivery)."""
        return self.days_between(self.as_of(), target)

    def _to_aware(self, value: _dt.datetime | _dt.date) -> _dt.datetime:
        if isinstance(value, _dt.datetime):
            return self.localize(value)  # type: ignore[return-value]
        # plain date -> midnight in presentation tz
        return _dt.datetime(value.year, value.month, value.day, tzinfo=self._tz)


def get_clock() -> Clock:
    """Build the clock from settings (cached settings -> cheap)."""
    from maxxflow_core.settings import get_settings

    s = get_settings()
    return Clock(presentation_tz=s.presentation_tz, as_of_override=s.as_of_override)
