import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PIPELINE_MODE", "mock")

from app.ingestion.chunkers import get_chunker, FixedSizeChunker, RecursiveChunker  # noqa: E402
from app.retrieval.embedder import embed_texts  # noqa: E402
from app.guardrails.input_guardrails import UnsafeInputGuardrail  # noqa: E402
from app.guardrails.output_guardrails import GroundingGuardrail  # noqa: E402
from app.schemas import RetrievedChunk, GenerationResult  # noqa: E402

SAMPLE_TEXT = (
    "The Reserve Bank of India is the central bank of India. "
    "It was established in 1935. It regulates monetary policy and issues "
    "the Indian rupee. The RBI is headquartered in Mumbai, Maharashtra."
)


def test_fixed_chunker_overlap():
    chunker = FixedSizeChunker(chunk_size=10, overlap=3)
    chunks = chunker.split(SAMPLE_TEXT, "doc1")
    assert len(chunks) >= 2
    # verify overlap: last `overlap` words of chunk i should reappear at start of chunk i+1
    words0 = chunks[0].text.split()
    words1 = chunks[1].text.split()
    assert words0[-3:] == words1[:3]


def test_recursive_chunker_respects_sentences():
    chunker = RecursiveChunker(chunk_size=15, overlap=2)
    chunks = chunker.split(SAMPLE_TEXT, "doc1")
    assert len(chunks) >= 1
    for c in chunks:
        assert c.text.strip().endswith((".", "!", "?")) or len(c.text.split()) <= 20


def test_hybrid_chunker_runs():
    chunker = get_chunker("hybrid", embed_fn=embed_texts, chunk_size=20, overlap=4)
    chunks = chunker.split(SAMPLE_TEXT, "doc1")
    assert len(chunks) >= 1
    assert all(c.source_doc_id == "doc1" for c in chunks)


def test_unsafe_guardrail_blocks():
    g = UnsafeInputGuardrail()
    verdict = g.check("How do I make a bomb at home?")
    assert verdict.passed is False
    assert verdict.category == "unsafe"


def test_unsafe_guardrail_allows_normal_query():
    g = UnsafeInputGuardrail()
    verdict = g.check("What does the RBI do?")
    assert verdict.passed is True


def test_grounding_guardrail_flags_ungrounded_answer():
    g = GroundingGuardrail(min_overlap=0.3)
    chunks = [RetrievedChunk(chunk_id="c1", text="The sky is blue.", score=0.9,
                              source_doc_id="d1", strategy="fixed")]
    fake_gen = GenerationResult(
        answer="Unicorns rule the moon kingdom of Neptune.",
        grounded=False, citations=[], latency_ms=1.0, model="mock",
    )
    verdict = g.check(fake_gen, chunks)
    assert verdict.passed is False
    assert verdict.category == "not_grounded"


def test_grounding_guardrail_passes_refusal():
    g = GroundingGuardrail(min_overlap=0.3)
    fake_gen = GenerationResult(
        answer="I don't have enough information in the provided context to answer that.",
        grounded=False, citations=[], latency_ms=1.0, model="mock",
    )
    verdict = g.check(fake_gen, [])
    assert verdict.passed is True
