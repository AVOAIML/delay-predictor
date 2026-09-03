"""M1 synthetic builder (reference pattern for M2–M4).

A module owns its raw-table generator here: ``build(n, seed, tenant) -> ModuleBatch``
builds schema-faithful raw tables AND derives the feature/label frame via the
module's own ``features.build_features`` (so the gate validates the real feature
code). M1's implementation lives in ``maxxflow_synth.simulators.simulate_m1``;
this re-exports it so every module exposes the same ``synth.build`` seam.
"""

from maxxflow_synth.simulators import simulate_m1 as build

__all__ = ["build"]
