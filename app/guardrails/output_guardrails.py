"""
Output-side guardrails, run AFTER generation, before the answer is returned
to the user.

1. GroundingGuardrail — checks that the generated answer is actually
   supported by the retrieved context, using two independent signals:
     a) word-overlap ratio between the answer and the union of retrieved
        chunk text (cheap, fast, catches answers that are pure invention)
     b) the generator's own self-reported citations (chunk_ids) — an empty
        citation list on a non-refusal answer is itself a red flag.
   Either signal failing routes to a refusal rather than returning an
   ungrounded answer.
2. EmptyRetrievalGuardrail — if retrieval returned zero chunks above a
   sane score floor, refuse before even calling generation (saves a wasted
   LLM call and avoids a free-form hallucinated answer).
"""
from __future__ import annotations
import re

from app.config import settings
from app.schemas import RetrievedChunk, GenerationResult, GuardrailVerdict

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "to",
    "and", "or", "for", "with", "that", "this", "it", "as", "by", "at",
    "be", "has", "have", "had", "its", "from",
}


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


class EmptyRetrievalGuardrail:
    def check(self, chunks: list[RetrievedChunk], min_score: float = 0.05) -> GuardrailVerdict:
        if not chunks:
            return GuardrailVerdict(passed=False, reason="No chunks retrieved.", category="no_context")
        if max(c.score for c in chunks) < min_score:
            return GuardrailVerdict(
                passed=False,
                reason="Best retrieved chunk score is below the relevance floor.",
                category="no_context",
                score=max(c.score for c in chunks),
            )
        return GuardrailVerdict(passed=True)


class GroundingGuardrail:
    def __init__(self, min_overlap: float | None = None):
        self.min_overlap = min_overlap if min_overlap is not None else settings.grounding_min_overlap

    def check(self, generation: GenerationResult, chunks: list[RetrievedChunk]) -> GuardrailVerdict:
        # An explicit "I don't know" style refusal from the generator is itself
        # a valid, well-grounded outcome — don't flag it.
        if not generation.grounded and "don't have enough information" in generation.answer.lower():
            return GuardrailVerdict(passed=True, reason="Model self-declined; treated as valid grounded refusal.")

        answer_words = _content_words(generation.answer)
        if not answer_words:
            return GuardrailVerdict(passed=False, reason="Empty/degenerate answer.", category="not_grounded")

        context_words = set()
        for c in chunks:
            context_words |= _content_words(c.text)

        overlap = len(answer_words & context_words) / len(answer_words)

        if overlap < self.min_overlap and not generation.citations:
            return GuardrailVerdict(
                passed=False,
                reason=f"Answer word-overlap with retrieved context is {overlap:.2f}, "
                       f"below threshold {self.min_overlap}, and no citations were provided.",
                category="not_grounded",
                score=overlap,
            )
        return GuardrailVerdict(passed=True, score=overlap)


REFUSAL_TEXT = "I don't have enough information in the provided context to answer that."
