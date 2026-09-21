"""Review Agent — judges whether each composed delay-insight line is
supported by the Risk Engine's evidence.

Public entry point: :class:`ReviewAgent`. See ``resolver.py`` for the single
judging call it wraps, and ``modules/m3_production_delay/review/README.md``
for the pipeline it sits inside.

The agent is asked yes/no questions and nothing else: :class:`JudgeVerdict`
has no field through which a model could return a number, a signal value, or
replacement wording. Every failure path — unreachable provider, unreadable
response, agent switched off — produces an unapproved verdict, never an
exception and never a default approval.
"""

from m3_production_delay.llm_agents.review_agent.config import (
    ReviewAgentConfig,
    build_config,
    get_review_agent_config,
)
from m3_production_delay.llm_agents.review_agent.exceptions import (
    JudgeResponseError,
    ReviewAgentError,
    ReviewConfigError,
)
from m3_production_delay.llm_agents.review_agent.models import (
    SKIPPED_VERDICT,
    JudgeLineVerdict,
    JudgeVerdict,
    coerce_diagnostics,
)
from m3_production_delay.llm_agents.review_agent.resolver import (
    AGENT_VERSION,
    PROMPT_VERSION,
    ReviewAgent,
    build_user_prompt,
    render_candidate_lines,
)
from m3_production_delay.llm_agents.review_agent.tracing import NOOP_TRACER, ReviewAgentTracer

__all__ = [
    "ReviewAgent",
    "ReviewAgentConfig",
    "build_config",
    "get_review_agent_config",
    "JudgeVerdict",
    "JudgeLineVerdict",
    "SKIPPED_VERDICT",
    "coerce_diagnostics",
    "ReviewAgentError",
    "ReviewConfigError",
    "JudgeResponseError",
    "ReviewAgentTracer",
    "NOOP_TRACER",
    "PROMPT_VERSION",
    "AGENT_VERSION",
    "build_user_prompt",
    "render_candidate_lines",
]
