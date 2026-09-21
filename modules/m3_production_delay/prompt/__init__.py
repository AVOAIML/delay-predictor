"""Prompt templates for M3's LLM-backed agents, kept as plain text files
rather than string literals in code — editing a prompt's wording never
requires touching the module that calls it. Each caller (profile_extractor.py,
weight_adjustment.py) loads its own template once at import time and fills it
in with ``string.Template`` (``$name`` placeholders, not ``{}``, since the
prompts themselves contain literal JSON braces).
"""

from __future__ import annotations

from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parent


def load_prompt_template(filename: str) -> str:
    return (_PROMPT_DIR / filename).read_text(encoding="utf-8")
