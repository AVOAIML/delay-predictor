"""Embedding providers. OFF by default — M4's default semantic path is classical
TF-IDF (char n-grams). No transformer is hosted either way (plan §2, §4 M4)."""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

_DIM = 256  # stub embedding dim (small; deterministic hashing trick)


class StubEmbeddingProvider:
    """Deterministic hashing-trick embeddings for offline/CI. ``enabled`` mirrors
    the ``EMBEDDINGS_ENABLED`` toggle so M4 only adds the embedding feature when
    explicitly switched on."""

    name = "stub"

    def __init__(self, enabled: bool = False, dim: int = _DIM):
        self._enabled = enabled
        self.dim = dim

    @property
    def enabled(self) -> bool:
        return self._enabled

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in str(t).lower().split():
                idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
                vecs[i, idx] += 1.0
            n = np.linalg.norm(vecs[i])
            if n > 0:
                vecs[i] /= n
        return vecs


class AzureOpenAIEmbeddingProvider:  # pragma: no cover - Phase 2 seam
    """Phase-2 adapter: text-embedding-3-small (AU East). VERIFY regional
    deployability BEFORE enabling (plan §12a #12)."""

    name = "azure_openai"

    def __init__(self, enabled: bool = False):
        raise NotImplementedError(
            "Azure OpenAI embeddings adapter is a Phase-2 seam. VERIFY "
            "text-embedding-3-small is deployable in AU East and set "
            "EMBEDDING_MODEL_ID before enabling."
        )

    @property
    def enabled(self) -> bool:
        return False

    def embed(self, texts):
        raise NotImplementedError
