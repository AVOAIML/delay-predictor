"""In-process event queue + micro-batch gating (local twin of Event Grid / Service
Bus). The envelope carries the authoritative payload + label to beat replica lag."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class EventEnvelope:
    topic: str
    tenant: str
    payload: Mapping[str, Any]          # authoritative snapshot (M3) / signal_vector (M4)
    label: Any = None                   # carried so it never depends on replica catch-up
    ts: float = field(default_factory=time.time)


class InProcEventQueue:
    """Implements :class:`maxxflow_core.ports.EventQueue`. Process-local, durable
    enough for the local inner loop; swapped for Service Bus / Event Grid in prod."""

    def __init__(self):
        self._topics: dict[str, list[EventEnvelope]] = {}

    def publish(self, topic: str, payload: Mapping[str, Any]) -> None:
        env = payload if isinstance(payload, EventEnvelope) else EventEnvelope(
            topic=topic, tenant=str(payload.get("tenant", "demo")), payload=payload,
            label=payload.get("label"))
        self._topics.setdefault(topic, []).append(env)

    def publish_envelope(self, env: EventEnvelope) -> None:
        self._topics.setdefault(env.topic, []).append(env)

    def peek(self, topic: str) -> list[EventEnvelope]:
        return list(self._topics.get(topic, []))

    def drain(self, topic: str) -> list[dict]:
        items = self._topics.pop(topic, [])
        return [{"tenant": e.tenant, "payload": dict(e.payload), "label": e.label, "ts": e.ts} for e in items]


_QUEUE = InProcEventQueue()


def get_event_queue() -> InProcEventQueue:
    return _QUEUE


def micro_batch_ready(events: list, *, min_count: int, max_age_seconds: float,
                      now: float | None = None) -> bool:
    """Debounce gate (plan §6): fire when ≥min_count events OR oldest ≥max_age.
    Never literal per-event training."""
    if not events:
        return False
    if len(events) >= min_count:
        return True
    now = now if now is not None else time.time()
    oldest = min(e.ts if isinstance(e, EventEnvelope) else e.get("ts", now) for e in events)
    return (now - oldest) >= max_age_seconds
