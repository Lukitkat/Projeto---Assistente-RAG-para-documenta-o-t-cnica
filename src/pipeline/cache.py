"""Cache em 2 níveis: exact-match (SHA256) + semantic (cosine similarity).

Usa fastembed (ONNX, sem PyTorch) para embeddings semânticos.
Fallback para sentence-transformers se fastembed não estiver disponível.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Callable

import numpy as np


def _build_embed_fn(model_name: str = "BAAI/bge-small-en-v1.5") -> Callable[[str], np.ndarray]:
    """Retorna função de embedding local (fastembed ou sentence-transformers).

    Retorna callable: str -> np.ndarray (vetor normalizado).
    """
    # Tentativa 1: fastembed (ONNX, sem PyTorch)
    try:
        from fastembed import TextEmbedding as FastTextEmbedding
        _model = FastTextEmbedding(model_name=model_name, cache_dir=".cache/fastembed", threads=1)

        def embed_fast(text: str) -> np.ndarray:
            vecs = list(_model.embed([text]))
            return np.array(vecs[0], dtype=np.float32)

        return embed_fast
    except Exception:
        pass

    # Tentativa 2: sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        st_name = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
        _st_model = SentenceTransformer(st_name)

        def embed_st(text: str) -> np.ndarray:
            vec = _st_model.encode(text, normalize_embeddings=True)
            return np.array(vec, dtype=np.float32)

        return embed_st
    except Exception:
        pass

    raise RuntimeError(
        "Nenhum provider de embeddings disponível para o SemanticCache. "
        "Instale o Microsoft Visual C++ Redistributable e reinicie o terminal."
    )


class ExactCache:
    """Cache por hash SHA256 da query. Captura replays exatos (~10-15% das queries)."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()

    def get(self, query: str) -> str | None:
        return self._store.get(self._key(query))

    def put(self, query: str, answer: str) -> None:
        self._store[self._key(query)] = answer

    def stats(self) -> dict[str, int]:
        return {"size": len(self._store)}


class SemanticCache:
    """Cache por similaridade de embedding. Captura paráfrases (~20% adicional).

    Usa fastembed (ONNX) — sem custo e sem necessidade de API key ou PyTorch.
    """

    def __init__(
        self,
        threshold: float = 0.93,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.threshold = threshold
        self._queries: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._answers: list[str] = []
        self._embed = _build_embed_fn(model_name=model_name)

    # ------------------------------------------------------------------ TODO 5
    def get(self, query: str) -> str | None:
        """Retorna resposta cacheada se similar a alguma query anterior, OU None."""
        if not self._queries:
            return None

        # get embedding for the current query
        query_emb = self._embed(query)

        # compute cosine similarity (dot product works since fastembed vectors are normalized)
        similarities = np.array([
            float(np.dot(query_emb, stored_emb))
            for stored_emb in self._embeddings
        ])

        # find the most similar stored query
        best_idx = int(np.argmax(similarities))

        # return the cached answer if similarity meets the threshold
        if similarities[best_idx] >= self.threshold:
            return self._answers[best_idx]

        return None

    def put(self, query: str, answer: str) -> None:
        self._queries.append(query)
        self._embeddings.append(self._embed(query))
        self._answers.append(answer)

    def stats(self) -> dict[str, Any]:
        return {"size": len(self._queries), "threshold": self.threshold}
