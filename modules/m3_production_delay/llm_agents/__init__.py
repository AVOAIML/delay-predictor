"""M3 LLM-backed agents.

``weight_agent`` resolves the tenant's risk-signal weight vector;
``review_agent`` judges whether each composed delay-insight line is supported
by the Risk Engine's evidence. Both are composed by
``m3_production_delay.orchestrator``, and both take the same optional
``llm_provider`` so one injected provider covers every LLM call this module
makes. ``eval_agent`` is a separate, not-yet-built component.
"""
