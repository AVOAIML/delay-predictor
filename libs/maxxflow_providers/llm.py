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
