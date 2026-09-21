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


class AzureAIFoundryLLMProvider:  # pragma: no cover - real network call, not exercised in CI
    """Azure AI Foundry adapter (``LLM_PROVIDER=azure_ai``), for the Foundry
    model catalogue — Anthropic Claude among it.

    This is a DIFFERENT data plane from :class:`AzureOpenAILLMProvider`, not a
    second spelling of it. Azure OpenAI serves OpenAI's own models from
    ``*.openai.azure.com`` with an ``api-version`` query parameter and its own
    key; Foundry serves the wider catalogue from its own endpoint with its own
    key. A Claude deployment is unreachable through the Azure OpenAI client no
    matter how the endpoint is written, which is why this class exists rather
    than another branch inside that one.

    Wire format is OpenAI-compatible Chat Completions, so the same ``openai``
    SDK drives it — pointed at ``AZURE_AI_API_BASE`` as a plain ``base_url``
    with a bearer key, no ``api_version``. Foundry publishes that route under
    ``/models`` on the resource endpoint; the suffix is appended here when the
    configured base does not already carry a path, so both spellings of
    ``AZURE_AI_API_BASE`` work:

        https://<resource>.services.ai.azure.com
        https://<resource>.services.ai.azure.com/models

    ``LLM_MODEL_ID`` must hold the **deployment name** chosen when the model
    was deployed, which is not necessarily the vendor's model name. Per the
    module's verify-before-building rule it is never defaulted or guessed:
    construction fails loudly if it is unset.
    """

    name = "azure_ai"

    #: Foundry's OpenAI-compatible route. Appended only when the configured
    #: base has no path of its own, so an already-complete endpoint is left
    #: exactly as the operator wrote it.
    _INFERENCE_PATH = "/models"

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.llm_model_id:
            raise NotImplementedError(
                "LLM_MODEL_ID is not set. Set it to the VERIFIED Azure AI Foundry "
                "DEPLOYMENT name (as shown in the Foundry deployment list, not the "
                "vendor's model name) before selecting LLM_PROVIDER=azure_ai."
            )
        api_key = settings.azure_ai_api_key.get_secret_value()
        if not api_key or not settings.azure_ai_api_base:
            raise NotImplementedError(
                "LLM_PROVIDER=azure_ai requires AZURE_AI_API_KEY and AZURE_AI_API_BASE. "
                "These are Foundry credentials — the AZURE_OPENAI_* settings belong to a "
                "different data plane and cannot serve a Foundry deployment."
            )
        self._model = settings.llm_model_id
        import openai

        self._client = openai.OpenAI(
            base_url=self.resolve_base_url(settings.azure_ai_api_base), api_key=api_key
        )

    @classmethod
    def resolve_base_url(cls, configured: str) -> str:
        """Normalise ``AZURE_AI_API_BASE`` to the inference route.

        A bare resource endpoint gets ``/models`` appended; anything that
        already carries a path is passed through untouched, so an operator
        who pasted the full route — or a future one that differs — is never
        second-guessed by this adapter.
        """
        base = configured.rstrip("/")
        scheme, _, remainder = base.partition("://")
        has_path = "/" in remainder if scheme else "/" in base
        return base if has_path else f"{base}{cls._INFERENCE_PATH}"

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
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
        except openai.BadRequestError as exc:
            # Foundry passes each vendor's own parameter rules through, and
            # they differ: `seed` in particular is an OpenAI concept that
            # non-OpenAI deployments reject outright. Retry once without the
            # parameters the deployment refused rather than failing every
            # call — the same best-effort stance the Azure OpenAI adapter
            # takes on `temperature`.
            if exc.param not in ("seed", "temperature"):
                raise
            request: dict = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }
            if exc.param != "temperature":
                request["temperature"] = temperature
            response = self._client.chat.completions.create(**request)
        return response.choices[0].message.content or ""
