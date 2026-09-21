"""Find out which API surface an Azure AI Foundry endpoint actually speaks.

Foundry fronts several different wire formats on one resource, and which one
you get depends on the path:

    .../models                 Azure AI Model Inference  (OpenAI-ish)
    .../openai/v1              OpenAI Chat Completions
    .../anthropic              Anthropic Messages        (/v1/messages)

They differ in route, auth header AND response shape, so an adapter written
for one returns 404 or unparsable output against another. Rather than guess,
this tries the plausible combinations and reports which returns 200.

Uses only the standard library — no openai or anthropic SDK — so it runs
before any of that is installed, and inside the container as-is.

    uv run python scripts/probe_foundry.py

Credentials come from .env.local / the environment, never from the command
line (shell history) and never from an argument. The key is never printed.
Paste the OUTPUT anywhere you like: it contains no secret.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

TIMEOUT = 30
PROMPT = "Reply with exactly: OK"


def _root(base: str) -> str:
    """The resource endpoint with any API path stripped off."""
    trimmed = base.rstrip("/")
    for suffix in ("/anthropic", "/models", "/openai/v1", "/openai"):
        if trimmed.endswith(suffix):
            return trimmed[: -len(suffix)]
    return trimmed


def _post(url: str, headers: dict, body: dict) -> tuple[int, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:  # DNS, TLS, timeout
        return 0, f"{type(exc).__name__}: {exc}"


def _messages_body(model: str) -> dict:
    return {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": PROMPT}],
    }


def _chat_body(model: str) -> dict:
    return {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": PROMPT}],
    }


def attempts(base: str, key: str, model: str) -> list[tuple[str, str, dict, dict]]:
    """(label, url, headers, body) for each combination worth trying."""
    root = _root(base)
    anthropic = f"{root}/anthropic"
    bare = model.split("/", 1)[-1]  # "azure_ai/claude-x" -> "claude-x"

    anthropic_versions = {"anthropic-version": "2023-06-01"}
    out: list[tuple[str, str, dict, dict]] = []

    # --- Anthropic Messages, the surface the /anthropic path implies --------
    for auth_label, auth in (
        ("x-api-key", {"x-api-key": key}),
        ("Authorization: Bearer", {"Authorization": f"Bearer {key}"}),
        ("api-key", {"api-key": key}),
    ):
        for model_label, model_value in (("bare model", bare), ("as configured", model)):
            out.append((
                f"Messages /v1/messages | {auth_label} | {model_label}",
                f"{anthropic}/v1/messages",
                {**auth, **anthropic_versions},
                _messages_body(model_value),
            ))

    # --- OpenAI-compatible surfaces, in case the resource fronts those too --
    for path_label, url in (
        ("models/chat/completions", f"{root}/models/chat/completions"),
        ("openai/v1/chat/completions", f"{root}/openai/v1/chat/completions"),
    ):
        for auth_label, auth in (
            ("Authorization: Bearer", {"Authorization": f"Bearer {key}"}),
            ("api-key", {"api-key": key}),
        ):
            out.append((
                f"OpenAI {path_label} | {auth_label} | as configured",
                url,
                auth,
                _chat_body(model),
            ))
    return out


def extract_text(payload: str) -> str | None:
    """Pull the assistant's text out of whichever response shape came back."""
    try:
        data = json.loads(payload)
    except Exception:
        return None
    # Anthropic Messages: {"content":[{"type":"text","text":"OK"}]}
    content = data.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text")
    # OpenAI: {"choices":[{"message":{"content":"OK"}}]}
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") or {}
        return message.get("content")
    return None


def main() -> int:
    from maxxflow_core.settings import get_settings

    settings = get_settings()
    base = settings.azure_ai_api_base
    key = settings.azure_ai_api_key.get_secret_value()
    model = settings.llm_model_id

    if not base or not key or not model:
        print("AZURE_AI_API_BASE, AZURE_AI_API_KEY and LLM_MODEL_ID must all be set "
              "in .env.local before probing.")
        return 1

    print(f"resource root : {_root(base)}")
    print(f"model id      : {model!r}   (bare: {model.split('/', 1)[-1]!r})")
    print(f"key           : set, {len(key)} chars, ends …{key[-4:]}")
    print()

    winners = []
    for label, url, headers, body in attempts(base, key, model):
        status, payload = _post(url, headers, body)
        text = extract_text(payload) if status == 200 else None
        if status == 200:
            winners.append((label, url, headers, body))
            print(f"  [200] {label}")
            print(f"        url    : {url}")
            print(f"        reply  : {(text or payload)[:80]!r}")
        else:
            detail = payload.strip().replace("\n", " ")[:110]
            print(f"  [{status or 'ERR':>3}] {label}")
            print(f"        {detail}")

    print()
    if winners:
        label, url, headers, _ = winners[0]
        print(f"WORKING: {label}")
        print(f"  route : {url}")
        print(f"  auth  : {', '.join(k for k in headers if k.lower() != 'content-type')}")
        print("\nPaste this whole output back and the adapter will be wired to match.")
        return 0

    print("Nothing answered 200. The statuses above say why — a 401/403 means the key or")
    print("its header is wrong, a 404 means the route is, and a 400 usually means the")
    print("model id is not a deployment on this resource.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
