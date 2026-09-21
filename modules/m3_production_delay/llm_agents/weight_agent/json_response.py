"""Strict helpers for parsing JSON returned by Weight Agent LLM stages."""

from __future__ import annotations


def unwrap_json_code_fence(raw_text: str) -> str:
    """Remove one complete Markdown JSON fence and nothing else.

    Models sometimes wrap an otherwise valid JSON document in a Markdown
    ``json`` code block.
    Accept that packaging, but keep surrounding prose and incomplete fences
    untouched so the caller's strict ``json.loads`` still rejects them.
    """
    text = raw_text.strip()
    lines = text.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        return text
    if lines[0].strip().lower() not in ("```", "```json"):
        return text
    return "\n".join(lines[1:-1]).strip()
