"""
The harness. This is the "more than a single raw prompt-in, text-out call"
layer: it sequences STT -> input guardrails -> retrieval -> output-guarded
generation, with per-stage:
  - structured Pydantic input/output (app/schemas.py)
  - bounded retries with backoff on transient stage failures
  - graceful error recovery (a stage failure produces a PipelineResponse
    with ok=False and a specific PipelineError, never an unhandled
    exception bubbling to the API layer)
  - short-circuiting refusals from guardrails, each carrying a
    human-readable reason back to the caller instead of a silent failure
  - full latency breakdown per stage, used by the benchmarking script
"""
from __future__ import annotations
import time

from app.guardrails.input_guardrails import UnsafeInputGuardrail, OffTopicGuardrail
from app.guardrails.output_guardrails import EmptyRetrievalGuardrail, GroundingGuardrail, REFUSAL_TEXT
from app.generation.generator import get_generator
from app.retrieval.retriever import Retriever
from app.schemas import (
    PipelineResponse, PipelineError, PipelineStage, TranscriptionResult,
)
from app.stt.base import BaseSTT


class RagVoiceOrchestrator:
    def __init__(self, stt: BaseSTT, retriever: Retriever, corpus_centroid=None):
        self.stt = stt
        self.retriever = retriever
        self.generator = get_generator()
        self.unsafe_guardrail = UnsafeInputGuardrail()
        self.offtopic_guardrail = OffTopicGuardrail(corpus_centroid)
        self.empty_retrieval_guardrail = EmptyRetrievalGuardrail()
        self.grounding_guardrail = GroundingGuardrail()

    def _run_stage(self, stage: PipelineStage, fn, *args, max_attempts: int = 2, **kwargs):
        """Generic retry wrapper for a pipeline stage. Returns (result, error)."""
        last_exc = None
        for attempt in range(1, max_attempts + 1):
            try:
                return fn(*args, **kwargs), None
            except Exception as e:  # noqa: BLE001 - intentional broad catch at harness boundary
                last_exc = e
                if attempt < max_attempts:
                    time.sleep(0.05 * attempt)  # tiny backoff, keeps us well under latency budget
                    continue
        return None, PipelineError(
            stage=stage,
            message=str(last_exc),
            recoverable=False,
        )

    def run(self, audio_bytes: bytes, language_hint: str | None = None, top_k: int = 5) -> PipelineResponse:
        t0 = time.perf_counter()
        latency: dict[str, float] = {}
        errors: list[PipelineError] = []

        # ---- Stage 1: Speech-to-Text ----
        transcription, err = self._run_stage(PipelineStage.STT, self.stt.transcribe, audio_bytes, language_hint)
        if err:
            errors.append(err)
            return self._finish(ok=False, errors=errors, latency=latency, t0=t0)
        transcription: TranscriptionResult
        latency["stt_ms"] = transcription.raw_latency_ms
        query_text = transcription.text.strip()

        if not query_text:
            errors.append(PipelineError(stage=PipelineStage.STT, message="Empty transcription.", recoverable=False))
            return self._finish(ok=False, errors=errors, latency=latency, t0=t0, transcript=query_text)

        # ---- Stage 2: Input guardrails ----
        stage_start = time.perf_counter()
        unsafe_verdict = self.unsafe_guardrail.check(query_text)
        if not unsafe_verdict.passed:
            latency["input_guardrail_ms"] = (time.perf_counter() - stage_start) * 1000
            return self._finish(
                ok=True, refused=True, refusal_reason=unsafe_verdict.reason,
                errors=errors, latency=latency, t0=t0, transcript=query_text,
            )

        offtopic_verdict = self.offtopic_guardrail.check(query_text)
        latency["input_guardrail_ms"] = (time.perf_counter() - stage_start) * 1000
        if not offtopic_verdict.passed:
            return self._finish(
                ok=True, refused=True, refusal_reason=offtopic_verdict.reason,
                errors=errors, latency=latency, t0=t0, transcript=query_text,
            )

        # ---- Stage 3: Retrieval ----
        retrieval_result, err = self._run_stage(PipelineStage.RETRIEVAL, self.retriever.retrieve, query_text, top_k)
        if err:
            errors.append(err)
            return self._finish(ok=False, errors=errors, latency=latency, t0=t0, transcript=query_text)
        latency["retrieval_ms"] = retrieval_result.latency_ms

        empty_verdict = self.empty_retrieval_guardrail.check(retrieval_result.chunks)
        if not empty_verdict.passed:
            return self._finish(
                ok=True, refused=True, refusal_reason=empty_verdict.reason,
                errors=errors, latency=latency, t0=t0, transcript=query_text,
                chunks=retrieval_result.chunks,
            )

        # ---- Stage 4: Generation ----
        generation_result, err = self._run_stage(
            PipelineStage.GENERATION, self.generator.generate, query_text, retrieval_result.chunks
        )
        if err:
            errors.append(err)
            return self._finish(
                ok=False, errors=errors, latency=latency, t0=t0,
                transcript=query_text, chunks=retrieval_result.chunks,
            )
        latency["generation_ms"] = generation_result.latency_ms

        # ---- Stage 5: Output guardrails (grounding / hallucination check) ----
        stage_start = time.perf_counter()
        grounding_verdict = self.grounding_guardrail.check(generation_result, retrieval_result.chunks)
        latency["output_guardrail_ms"] = (time.perf_counter() - stage_start) * 1000

        if not grounding_verdict.passed:
            return self._finish(
                ok=True, refused=True, refusal_reason=grounding_verdict.reason,
                errors=errors, latency=latency, t0=t0, transcript=query_text,
                chunks=retrieval_result.chunks,
            )

        return self._finish(
            ok=True,
            answer=generation_result.answer,
            transcript=query_text,
            chunks=retrieval_result.chunks,
            errors=errors,
            latency=latency,
            t0=t0,
        )

    @staticmethod
    def _finish(*, ok, latency, t0, errors=None, transcript=None, chunks=None,
                answer=None, refused=False, refusal_reason=None) -> PipelineResponse:
        total_ms = (time.perf_counter() - t0) * 1000
        return PipelineResponse(
            ok=ok,
            answer=REFUSAL_TEXT if refused else answer,
            transcript=transcript,
            retrieved_chunks=chunks or [],
            refused=refused,
            refusal_reason=refusal_reason,
            errors=errors or [],
            latency_breakdown_ms=latency,
            total_latency_ms=round(total_ms, 3),
        )
