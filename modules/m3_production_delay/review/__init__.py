"""M3 Section 3 — Review Agent & Validated AI Insight.

The LLM-as-a-judge layer over the Risk Engine's output: deterministic evidence
from a scored job, template explanation lines, deterministic validators, and
then one LLM call that is only ever asked a yes/no question about whether each
line is supported by that evidence.

No number, signal or line of user-facing text is ever produced by the LLM.
See ``review/README.md`` for the contract and how to run one job end to end.
"""
