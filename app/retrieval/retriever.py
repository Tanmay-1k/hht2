"""
Retriever: wraps the VectorStore with a light reranking pass that combines
dense cosine similarity with metadata-derived signals (chunk position in
doc, chunk length) — pure ANN top-k is a good recall mechanism but a poor
precision mechanism on its own for short, boilerplate-heavy passages, so we
nudge the ranking rather than trusting raw cosine score alone.
"""
from __future__ import annotations
import time

from app.retrieval.vector_store import VectorStore
from app.schemas import RetrievedChunk, RetrievalResult


class Retriever:
    def __init__(self, store: VectorStore, top_k: int = 5):
        self.store = store
        self.top_k = top_k

    @staticmethod
    def _rerank_score(cosine_score: float, meta: dict) -> float:
        score = cosine_score
        # small boost for chunks near the start of a document (often more
        # information-dense / topic-defining in MS MARCO-style passages)
        pos_ratio = meta.get("position_ratio", 0.5)
        score += 0.03 * (1 - pos_ratio)
        # mild penalty for very short chunks (likely low-information fragments)
        n_words = meta.get("n_words", 100)
        if n_words < 15:
            score -= 0.05
        return score

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        start = time.perf_counter()
        k = top_k or self.top_k
        raw_hits = self.store.search(query, k=max(k * 3, k))  # over-fetch for reranking headroom

        reranked = sorted(
            raw_hits,
            key=lambda pair: self._rerank_score(pair[1], pair[0]),
            reverse=True,
        )[:k]

        chunks = [
            RetrievedChunk(
                chunk_id=meta["chunk_id"],
                text=meta["text"],
                score=round(score, 4),
                source_doc_id=meta["source_doc_id"],
                strategy=meta["strategy"],
                metadata=meta.get("metadata", {}),
            )
            for meta, score in reranked
        ]
        elapsed_ms = (time.perf_counter() - start) * 1000
        return RetrievalResult(query=query, chunks=chunks, latency_ms=elapsed_ms)
