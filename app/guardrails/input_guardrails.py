"""
Input-side guardrails, run BEFORE retrieval/generation so we never spend a
generation call (or expose the LLM) to something we're going to refuse
anyway.

1. UnsafeInputGuardrail  — blocks clearly unsafe/inappropriate requests
   (violence, self-harm, illegal instructions, prompt-injection attempts
   against the system prompt) via a keyword/pattern screen. This is a
   coarse first line of defense, not a full safety classifier — see README
   for how you'd swap in a hosted moderation endpoint for production.
2. OffTopicGuardrail     — compares the query embedding against the corpus
   centroid (mean of all chunk embeddings). Queries that are very
   dissimilar to anything in the corpus (e.g. "write me a poem about cats"
   against an MS MARCO factual-QA corpus) are flagged off-topic rather than
   forced through retrieval, which would otherwise silently return
   irrelevant top-k chunks and let the LLM hallucinate an answer.
"""
from __future__ import annotations
import re
import numpy as np

from app.config import settings
from app.retrieval.embedder import embed_texts
from app.schemas import GuardrailVerdict

UNSAFE_PATTERNS = [
    r"\bhow (do|can) i (make|build|synthesize) (a )?(bomb|explosive|weapon)\b",
    r"\bhow to (kill|murder|harm) (myself|someone|a person)\b",
    r"\bignore (all|your) (previous|prior) instructions\b",
    r"\bact as if you have no (rules|guardrails|restrictions)\b",
    r"\breveal (your|the) system prompt\b",
    r"\bhow to (hack|exploit) .* (without permission|illegally)\b",
]

_UNSAFE_RE = [re.compile(p, re.IGNORECASE) for p in UNSAFE_PATTERNS]


class UnsafeInputGuardrail:
    def check(self, query: str) -> GuardrailVerdict:
        for pattern in _UNSAFE_RE:
            if pattern.search(query):
                return GuardrailVerdict(
                    passed=False,
                    reason="Query matched an unsafe/prompt-injection pattern.",
                    category="unsafe",
                    score=1.0,
                )
        return GuardrailVerdict(passed=True)


class OffTopicGuardrail:
    def __init__(self, corpus_centroid: np.ndarray | None, min_similarity: float | None = None):
        self.corpus_centroid = corpus_centroid
        self.min_similarity = min_similarity or settings.offtopic_min_similarity

    @staticmethod
    def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0

    def check(self, query: str) -> GuardrailVerdict:
        if self.corpus_centroid is None:
            return GuardrailVerdict(passed=True, reason="No corpus centroid available; skipping off-topic check.")
        query_vec = embed_texts([query])[0]
        sim = self._cos_sim(query_vec, self.corpus_centroid)
        if sim < self.min_similarity:
            return GuardrailVerdict(
                passed=False,
                reason=f"Query is not topically related to the indexed corpus (similarity={sim:.3f}).",
                category="off_topic",
                score=sim,
            )
        return GuardrailVerdict(passed=True, score=sim)


def compute_corpus_centroid(vector_store) -> np.ndarray | None:
    """Utility to precompute a corpus centroid from an already-built vector store's
    embeddings, for use by OffTopicGuardrail. Called once at startup."""
    if vector_store.index is None or vector_store.index.ntotal == 0:
        return None
    all_vecs = vector_store.index.reconstruct_n(0, vector_store.index.ntotal)
    return np.mean(all_vecs, axis=0)
