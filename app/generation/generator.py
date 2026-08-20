"""
Generation layer. Abstracts away the LLM provider so the harness can call
`generate(question, chunks)` regardless of whether we're in mock mode or
hitting a real API. The mock generator does simple extractive answering
over the retrieved chunks so the pipeline is fully testable/benchmarkable
without any API key.
"""
from __future__ import annotations
import re
import time

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings
from app.generation.prompts import SYSTEM_PROMPT, ANSWER_TEMPLATE, build_context_block
from app.schemas import RetrievedChunk, GenerationResult

NO_ANSWER_TEXT = "I don't have enough information in the provided context to answer that."


class BaseGenerator:
    def generate(self, question: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        raise NotImplementedError


class MockGenerator(BaseGenerator):
    """Extractive fallback: picks the highest-scoring chunk(s) and returns
    the most relevant sentence(s) as the 'answer', so grounding is trivially
    guaranteed (the answer text is literally a substring of retrieved
    context) — useful for offline testing of the harness/guardrails."""

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        start = time.perf_counter()
        if not chunks:
            elapsed = (time.perf_counter() - start) * 1000
            return GenerationResult(answer=NO_ANSWER_TEXT, grounded=False, citations=[], latency_ms=elapsed, model="mock")

        best = chunks[0]
        sentences = re.split(r"(?<=[.!?])\s+", best.text)
        answer = " ".join(sentences[:2]).strip() or best.text[:280]
        elapsed = (time.perf_counter() - start) * 1000
        return GenerationResult(
            answer=answer,
            grounded=True,
            citations=[best.chunk_id],
            latency_ms=elapsed,
            model="mock-extractive",
        )


class AnthropicGenerator(BaseGenerator):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_model
        if not self.api_key:
            raise ValueError("LLM_API_KEY is not set for Anthropic provider.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.3, min=0.3, max=3),
        reraise=True,
    )
    def _call(self, question: str, chunks: list[RetrievedChunk]):
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        prompt = ANSWER_TEMPLATE.format(
            context_block=build_context_block(chunks),
            question=question,
        )
        resp = client.messages.create(
            model=self.model,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        start = time.perf_counter()
        try:
            raw = self._call(question, chunks)
        except Exception as e:
            raise RuntimeError(f"Anthropic generation failed after retries: {e}") from e
        elapsed = (time.perf_counter() - start) * 1000

        answer_match = re.search(r"ANSWER:\s*(.*?)\s*CITED_CHUNK_IDS:", raw, re.DOTALL)
        cite_match = re.search(r"CITED_CHUNK_IDS:\s*(.*)", raw, re.DOTALL)
        answer = (answer_match.group(1).strip() if answer_match else raw.strip())
        citations_raw = cite_match.group(1).strip() if cite_match else "NONE"
        citations = [] if citations_raw.upper() == "NONE" else [c.strip() for c in citations_raw.split(",") if c.strip()]

        return GenerationResult(
            answer=answer,
            grounded=bool(citations),
            citations=citations,
            latency_ms=elapsed,
            model=self.model,
        )


class SarvamGenerator(BaseGenerator):
    """
    Uses Sarvam's Chat Completion API (OpenAI-compatible /v1/chat/completions
    shape) for generation, so the same SARVAM_API_KEY used for STT also
    covers the LLM call — no separate paid provider needed to get a live
    demo running. Docs: https://docs.sarvam.ai/api-reference-docs/chat-completion/overview
    """
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.sarvam_api_key
        self.model = model or (settings.llm_model if "sarvam" in settings.llm_model else "sarvam-105b")
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY is not set for Sarvam LLM provider.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.3, min=0.3, max=3),
        reraise=True,
    )
    def _call(self, question: str, chunks: list[RetrievedChunk]) -> str:
        import httpx
        prompt = ANSWER_TEMPLATE.format(
            context_block=build_context_block(chunks),
            question=question,
        )
        headers = {"api-subscription-key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 400,
            "temperature": 0.2,
        }
        with httpx.Client(timeout=15.0) as client:
            resp = client.post("https://api.sarvam.ai/v1/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        start = time.perf_counter()
        try:
            raw = self._call(question, chunks)
        except Exception as e:
            raise RuntimeError(f"Sarvam generation failed after retries: {e}") from e
        elapsed = (time.perf_counter() - start) * 1000

        answer_match = re.search(r"ANSWER:\s*(.*?)\s*CITED_CHUNK_IDS:", raw, re.DOTALL)
        cite_match = re.search(r"CITED_CHUNK_IDS:\s*(.*)", raw, re.DOTALL)
        answer = (answer_match.group(1).strip() if answer_match else raw.strip())
        citations_raw = cite_match.group(1).strip() if cite_match else "NONE"
        citations = [] if citations_raw.upper() == "NONE" else [c.strip() for c in citations_raw.split(",") if c.strip()]

        return GenerationResult(
            answer=answer,
            grounded=bool(citations),
            citations=citations,
            latency_ms=elapsed,
            model=self.model,
        )


def get_generator() -> BaseGenerator:
    if settings.pipeline_mode == "mock" or settings.llm_provider == "mock":
        return MockGenerator()
    if settings.llm_provider == "sarvam":
        return SarvamGenerator()
    if settings.llm_provider == "anthropic":
        return AnthropicGenerator()
    raise ValueError(f"Unknown/unimplemented LLM_PROVIDER: {settings.llm_provider}")
