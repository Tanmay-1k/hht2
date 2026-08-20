"""
Structured I/O contracts. Every stage of the pipeline consumes and produces
one of these models instead of raw dicts/strings — this is what the harness
uses to validate stage outputs and decide whether to retry / fall back.
"""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class PipelineStage(str, Enum):
    STT = "speech_to_text"
    INPUT_GUARDRAIL = "input_guardrail"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    OUTPUT_GUARDRAIL = "output_guardrail"


class TranscriptionResult(BaseModel):
    text: str
    language: str | None = None
    confidence: float | None = None
    provider: str
    raw_latency_ms: float


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    source_doc_id: str
    strategy: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    latency_ms: float


class GuardrailVerdict(BaseModel):
    passed: bool
    reason: str | None = None
    category: str | None = None  # e.g. "off_topic", "unsafe", "not_grounded"
    score: float | None = None


class GenerationResult(BaseModel):
    answer: str
    grounded: bool
    citations: list[str] = Field(default_factory=list)
    latency_ms: float
    model: str


class PipelineError(BaseModel):
    stage: PipelineStage
    message: str
    recoverable: bool


class PipelineResponse(BaseModel):
    """Final structured response returned by the /query endpoint."""
    ok: bool
    answer: str | None = None
    transcript: str | None = None
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    errors: list[PipelineError] = Field(default_factory=list)
    latency_breakdown_ms: dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float = 0.0
