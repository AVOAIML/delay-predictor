"""Check that the configured LLM provider actually answers, before spending a
batch run finding out that it does not.

Reports what `Settings` resolved, constructs the provider through the normal
factory, and makes ONE tiny call. It never prints a key — only whether one is
present and how it ends — so the output is safe to paste into a chat or a
ticket when something is wrong.

    uv run python scripts/check_llm.py            # whatever .env.local selects
    uv run python scripts/check_llm.py --judge    # also run one real verdict

Exit code is 0 only if the provider answered.
"""

from __future__ import annotations

import argparse
import json
import time


def _redact(secret: str) -> str:
    """Presence and a 4-char tail — enough to tell two keys apart, useless to
    anyone who reads it."""
    if not secret:
        return "(not set)"
    return f"set, {len(secret)} chars, ends …{secret[-4:]}"


def describe() -> dict:
    from maxxflow_core.settings import get_settings

    settings = get_settings()
    return {
        "APP_ENV": settings.app_env,
        "LLM_PROVIDER": settings.llm_provider,
        "LLM_MODEL_ID": settings.llm_model_id or "(not set)",
        "AZURE_AI_API_BASE": settings.azure_ai_api_base or "(not set)",
        "AZURE_AI_API_KEY": _redact(settings.azure_ai_api_key.get_secret_value()),
        "M3_REVIEW_AGENT_LLM_ENABLED": settings.m3_review_llm_enabled,
    }


def ping() -> tuple[bool, str]:
    """One minimal generate() through the real factory."""
    from maxxflow_core.ports import GenerationConfig
    from maxxflow_providers import get_llm_provider

    provider = get_llm_provider()
    name = getattr(provider, "name", type(provider).__name__)
    started = time.perf_counter()
    reply = provider.generate(
        'Reply with exactly this JSON and nothing else: {"ok": true}',
        max_tokens=32,
        generation_config=GenerationConfig(temperature=0.0, seed=0),
    )
    latency = int((time.perf_counter() - started) * 1000)

    preview = (reply or "").strip().replace("\n", " ")[:120]
    print(f"  provider responded : {name} in {latency} ms")
    print(f"  reply              : {preview!r}")

    try:
        json.loads(reply)
    except Exception:
        # The stub always lands here: it returns prose, which is exactly why
        # it can never satisfy the judge.
        return False, (
            "the reply is not JSON — a judge call against this provider would parse-fail "
            "and publish `fallback_template`"
        )
    return True, "the reply parsed as JSON — the judge can work against this provider"


def judge_once() -> None:
    """One real verdict over the recorded fixture, end to end."""
    import pathlib

    from m3_production_delay.llm_agents.review_agent import ReviewAgent
    from m3_production_delay.review.composer import compose
    from m3_production_delay.review.evidence import build_evidence

    fixture = (
        pathlib.Path(__file__).resolve().parents[1]
        / "modules/m3_production_delay/review/fixtures/section1_job.json"
    )
    weights = {
        "time_overrun_ratio": 0.40,
        "operator_pace_ratio": 0.30,
        "material_shortfall_ratio": 0.15,
        "supplier_reliability": 0.05,
    }
    pack = build_evidence(json.loads(fixture.read_text(encoding="utf-8")), weights, 1.0)
    draft = compose(pack)

    print(f"\n  judging {len(draft.why_lines)} candidate lines for {pack.job_id} …")
    verdict = ReviewAgent().judge(pack, draft)
    print(f"  approved           : {verdict.approved}")
    print(f"  parse_error        : {verdict.parse_error}")
    if verdict.lines:
        for line in verdict.lines:
            mark = "supported" if line.supported else "NOT supported"
            print(f"    line {line.line_index}: {mark}  {line.issue or ''}")
    if verdict.omitted_signals:
        print(f"  omitted_signals    : {list(verdict.omitted_signals)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_llm", description="Verify the configured LLM provider answers."
    )
    parser.add_argument(
        "--judge", action="store_true", help="also run one real judging call over the fixture"
    )
    args = parser.parse_args(argv)

    print("Resolved configuration:")
    for key, value in describe().items():
        print(f"  {key:28} {value}")
    print()

    try:
        ok, message = ping()
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        print("\n  Common causes:")
        print("   - LLM_MODEL_ID is not the Foundry DEPLOYMENT name")
        print("   - AZURE_AI_API_BASE points at an *.openai.azure.com host (that is")
        print("     Azure OpenAI, a different data plane, and cannot serve Claude)")
        print("   - the key belongs to a different resource than the base URL")
        return 1

    print(f"\n  {message}")
    if args.judge and ok:
        judge_once()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
