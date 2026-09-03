"""maxxflow_events — local stand-in for Event Grid / Service Bus (plan §6, §10).

M3 (MO→Done) and M4 (confirmed correction) retrains are event-triggered. Locally
we use an in-process queue / APScheduler-style worker; the Azure adapter (Event
Grid event trigger for M3, Service Bus session-per-tenant + DLQ for M4) drops in
behind the same ``EventQueue`` port in Phase 2.

Replica-lag discipline (§12a #3): the authoritative payload — M3's completion
snapshot, M4's frozen signal_vector — AND the label are carried IN the envelope,
so the irreplaceable datum never waits on async replica catch-up.
"""

from maxxflow_events.queue import (
    EventEnvelope,
    InProcEventQueue,
    get_event_queue,
    micro_batch_ready,
)

__all__ = ["EventEnvelope", "InProcEventQueue", "get_event_queue", "micro_batch_ready"]
