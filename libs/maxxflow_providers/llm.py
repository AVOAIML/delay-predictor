"""LLM providers. Text generation ONLY — never scoring (plan §4 M4, §2)."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from maxxflow_core.ports import GenerationConfig
from maxxflow_core.settings import get_settings

if TYPE_CHECKING:
    import openai as openai_types


class StubLLMProvider:
    """Deterministic offline stub. Phrases a human-readable fix explanation from
    the structured fix already produced by the rule engine. No network, no PII."""

    name = "stub"

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        generation_config: GenerationConfig | None = None,
    ) -> str:
        # Already fully deterministic (content-addressed hash) regardless of
        # generation_config — accepted for Protocol conformance, not used.
        h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        first_line = prompt.strip().splitlines()[0] if prompt.strip() else ""
        return f"[stub-llm:{h}] Suggested explanation for: {first_line[:200]}"


class AzureOpenAILLMProvider:  # pragma: no cover - real network call, not exercised in CI
    """OpenAI-compatible Chat Completions adapter — serves both plain OpenAI
    (``LLM_PROVIDER=openai``, ``OPENAI_API_KEY``) and Azure OpenAI
    (``LLM_PROVIDER=azure_openai``, ``AZURE_OPENAI_API_KEY`` +
    ``AZURE_OPENAI_ENDPOINT``), selected by ``Settings.llm_provider`` — same
    wire format either way, only the client/auth differs (factory.py already
    routed both names here). Requires a VERIFIED model/deployment id in
    ``LLM_MODEL_ID`` — never guessed. Construction fails loudly, not silently,
    if the required credentials for whichever provider was selected are
    missing, so a misconfigured deployment cannot masquerade as a working one.
    """

    name = "azure_openai"

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.llm_model_id:
            raise NotImplementedError(
                "LLM_MODEL_ID is not set. Set it to a VERIFIED model/deployment "
                "id before selecting a real LLM_PROVIDER."
            )
        self._model = settings.llm_model_id
        self._client: openai_types.OpenAI | openai_types.AzureOpenAI
        if settings.llm_provider == "openai":
            api_key = settings.openai_api_key.get_secret_value()
            if not api_key:
                raise NotImplementedError("LLM_PROVIDER=openai requires OPENAI_API_KEY.")
            import openai

            self._client = openai.OpenAI(api_key=api_key)
        else:
            api_key = settings.azure_openai_api_key.get_secret_value()
            if not api_key or not settings.azure_openai_endpoint:
                raise NotImplementedError(
                    "LLM_PROVIDER=azure_openai requires AZURE_OPENAI_API_KEY and "
                    "AZURE_OPENAI_ENDPOINT."
                )
            import openai

            self._client = openai.AzureOpenAI(
                api_key=api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version or "2024-08-01-preview",
            )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        generation_config: GenerationConfig | None = None,
    ) -> str:
        import openai

        temperature = generation_config.temperature if generation_config else 0.0
        seed = generation_config.seed if generation_config else None
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
        except openai.BadRequestError as exc:
            if exc.param != "temperature":
                raise
            # Verified empirically (not guessed): some models — gpt-5 among
            # them — only support their own default temperature and reject
            # any explicit value, including 0. Determinism-via-temperature is
            # a best-effort request to the provider, not something this port
            # can force universally, so retry once without it rather than
            # failing every call against such a model.
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                seed=seed,
            )
        return response.choices[0].message.content or ""


class AzureAIFoundryLLMProvider:
    """Azure AI Foundry adapter (``LLM_PROVIDER=azure_ai``) — the data plane
    that serves Foundry's non-OpenAI catalogue, Anthropic Claude among it.

    Foundry is NOT Azure OpenAI. It has its own endpoint and its own key, so a
    Claude deployment is unreachable through ``AZURE_OPENAI_*`` however the
    endpoint is spelled — which is why this exists rather than another branch
    inside :class:`AzureOpenAILLMProvider`.

    One resource fronts several wire formats, chosen by the path in
    ``AZURE_AI_API_BASE``, and they differ in route, request body AND response
    shape::

        .../anthropic       Anthropic Messages   POST /v1/messages
        .../models          Model Inference      POST /chat/completions
        .../openai/v1       OpenAI               POST /chat/completions

    So the surface is derived from the endpoint the operator configured rather
    than separately declared: the path IS the protocol, and asking them to
    state it twice would only create a way to disagree with themselves.
    Verified empirically against a staging resource — the Anthropic route
    answers on ``x-api-key`` and rejects an OpenAI-style ``api-key`` with 401.

    Deliberately built on ``urllib`` rather than the ``openai`` or
    ``anthropic`` SDK. Both surfaces are one POST of plain JSON, and the
    runtime image installs neither SDK (it syncs ``serve``/``ui``/``aml``, not
    ``llm``), so an SDK-based adapter fails with ``ModuleNotFoundError`` in
    exactly the container that needs it.

    ``LLM_MODEL_ID`` must be the DEPLOYMENT name as the Foundry deployment list
    shows it — a routing-prefixed name like ``azure_ai/claude-sonnet-4-6``
    returns ``DeploymentNotFound``. Never guessed: construction fails loudly
    when it, or either credential, is unset.
    """

    name = "azure_ai"

    #: Pinned, not floated: the Messages API is versioned by this header, and
    #: letting it drift would change the response shape without a code change.
    ANTHROPIC_VERSION = "2023-06-01"

    #: Foundry's OpenAI-compatible route, appended only when the configured
    #: base carries no path of its own.
    _INFERENCE_PATH = "/models"

    def __init__(self, *, timeout: float = 60.0) -> None:
        settings = get_settings()
        if not settings.llm_model_id:
            raise NotImplementedError(
                "LLM_MODEL_ID is not set. Set it to the VERIFIED Azure AI Foundry "
                "DEPLOYMENT name (as the Foundry deployment list shows it — not the "
                "vendor's model name, and not a routing-prefixed one) before selecting "
                "LLM_PROVIDER=azure_ai."
            )
        api_key = settings.azure_ai_api_key.get_secret_value()
        if not api_key or not settings.azure_ai_api_base:
            raise NotImplementedError(
                "LLM_PROVIDER=azure_ai requires AZURE_AI_API_KEY and AZURE_AI_API_BASE. "
                "These are Foundry credentials — the AZURE_OPENAI_* settings belong to a "
                "different data plane and cannot serve a Foundry deployment."
            )
        self._model = settings.llm_model_id
        self._api_key = api_key
        self._base_url = self.resolve_base_url(settings.azure_ai_api_base)
        self._timeout = timeout

    # --- endpoint -----------------------------------------------------------

    @classmethod
    def resolve_base_url(cls, configured: str) -> str:
        """Normalise ``AZURE_AI_API_BASE``.

        A bare resource endpoint gets Foundry's ``/models`` route appended;
        anything already carrying a path passes through untouched, so an
        operator who pasted ``/anthropic`` — or a future route — is never
        second-guessed.
        """
        base = configured.rstrip("/")
        scheme, _, remainder = base.partition("://")
        has_path = "/" in remainder if scheme else "/" in base
        return base if has_path else f"{base}{cls._INFERENCE_PATH}"

    @staticmethod
    def is_anthropic_surface(base_url: str) -> bool:
        """Whether this endpoint speaks Anthropic Messages rather than Chat
        Completions. The path is the protocol — see the class docstring."""
        return "/anthropic" in base_url.rstrip("/").lower()

    # --- request shaping (pure, so it is testable without a network) ---------

    def build_request(
        self, prompt: str, *, max_tokens: int, temperature: float, seed: int | None
    ) -> tuple[str, dict[str, str], dict]:
        """``(url, headers, body)`` for one generation on whichever surface
        this endpoint speaks."""
        if self.is_anthropic_surface(self._base_url):
            return (
                f"{self._base_url}/v1/messages",
                {
                    "content-type": "application/json",
                    "x-api-key": self._api_key,
                    "anthropic-version": self.ANTHROPIC_VERSION,
                },
                {
                    "model": self._model,
                    # Required by the Messages API, unlike Chat Completions.
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    # No `seed`: the Messages API has no such parameter and
                    # rejects unknown fields, so sending one would fail every
                    # call. Determinism here rests on temperature 0 plus the
                    # structural validation callers already do on the response.
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
        return (
            f"{self._base_url}/chat/completions",
            {"content-type": "application/json", "Authorization": f"Bearer {self._api_key}"},
            {
                "model": self._model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **({"seed": seed} if seed is not None else {}),
                "messages": [{"role": "user", "content": prompt}],
            },
        )

    @staticmethod
    def extract_text(payload: dict) -> str:
        """Pull the assistant's text out of either response shape.

        Anthropic returns ``content`` as a list of typed blocks, which may hold
        several text blocks (or none at all); OpenAI returns a single
        ``choices[].message.content`` string.
        """
        content = payload.get("content")
        if isinstance(content, list):
            return "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            return (choices[0].get("message") or {}).get("content") or ""
        return ""

    # --- the call -----------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        generation_config: GenerationConfig | None = None,
    ) -> str:
        import json
        import urllib.error
        import urllib.request

        temperature = generation_config.temperature if generation_config else 0.0
        seed = generation_config.seed if generation_config else None
        url, headers, body = self.build_request(
            prompt, max_tokens=max_tokens, temperature=temperature, seed=seed
        )

        request = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            detail = exc.read().decode("utf-8", "replace")[:400]
            # The status plus the provider's own message is what distinguishes a
            # wrong deployment name from a wrong key from a wrong route, so both
            # are surfaced rather than collapsed into a generic failure.
            raise RuntimeError(
                f"Azure AI Foundry returned {exc.code} for model {self._model!r} "
                f"at {url}: {detail}"
            ) from exc
        return self.extract_text(payload)


