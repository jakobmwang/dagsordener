"""Flagserve embedder for dense + sparse vectors."""

import os
from typing import Any

import httpx

FLAGSERVE_URL = os.getenv("FLAGSERVE_URL", "http://flagserve:8273")
FLAGSERVE_API_KEY = os.getenv("FLAGSERVE_API_KEY", "")

def _normalize_sparse(entry: Any) -> dict:
    if isinstance(entry, dict) and "indices" in entry and "values" in entry:
        return {"indices": entry["indices"], "values": entry["values"]}
    if isinstance(entry, dict):
        return {"indices": list(entry.keys()), "values": list(entry.values())}
    raise ValueError("Unsupported sparse format")


def _normalize_embed_response(data: dict, count: int) -> dict:
    if "dense_vecs" not in data or "lexical_weights" not in data:
        raise ValueError("Embed response missing dense_vecs/lexical_weights outputs")
    dense = data["dense_vecs"]
    sparse = data["lexical_weights"]
    if not isinstance(dense, list):
        raise ValueError("Embed response dense_vecs must be a list")
    if not isinstance(sparse, list):
        sparse = [sparse]

    normalized_sparse = [_normalize_sparse(item) for item in sparse]
    if count and len(dense) != count:
        raise ValueError("Dense output count mismatch")
    if count and len(normalized_sparse) != count:
        raise ValueError("Sparse output count mismatch")

    return {"dense": dense, "sparse": normalized_sparse}


def _normalize_rerank_response(data: Any, count: int) -> list[tuple[int, float]]:
    if not isinstance(data, dict):
        raise ValueError("Rerank response must be an object with scores")
    scores = data.get("scores")
    if not isinstance(scores, list):
        raise ValueError("Rerank response missing scores")
    if len(scores) != count:
        raise ValueError("Rerank score count mismatch")
    pairs = [(i, float(score)) for i, score in enumerate(scores)]
    return sorted(pairs, key=lambda item: item[1], reverse=True)


class Embedder:
    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 30.0,
    ):
        self.url = (url or FLAGSERVE_URL).rstrip("/")
        self.api_key = FLAGSERVE_API_KEY if api_key is None else api_key
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {
            "X-API-Key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
        }

    def embed(self, texts: list[str]) -> dict:
        """Embed texts and return dense + sparse vectors."""
        response = httpx.post(
            f"{self.url}/embed",
            json={
                "texts": texts,
                "return_dense": True,
                "return_sparse": True,
                "return_colbert": False,
            },
            headers=self._headers(),
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        return _normalize_embed_response(data, len(texts))

    def embed_single(self, text: str) -> dict:
        """Embed a single text."""
        result = self.embed([text])
        return {
            "dense": result["dense"][0],
            "sparse": result["sparse"][0],
        }

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        """Rerank documents for query. Returns (index, score) pairs."""
        if not documents:
            return []
        response = httpx.post(
            f"{self.url}/rerank",
            json={"query": query, "documents": documents},
            headers=self._headers(),
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        return _normalize_rerank_response(data, len(documents))
