"""maxxflow_synth — cold-start synthetic engine (plan §1, §5).

Domain-driven, seedable simulators (NOT SDV/CTGAN — there is no seed data to learn
from). Each simulator (a) respects the schema, (b) embeds a KNOWN latent
ground-truth function per module, (c) injects realistic noise / class imbalance /
seasonality. It writes schema-faithful, label-bearing rows into the tenant
Postgres schema (the same path production reads), and the 3 CI gates
(schema / realism / leakage+learnability) must pass before any batch trains.
"""

from maxxflow_synth.simulators import simulate

__all__ = ["simulate"]
