# Voice-Enabled RAG — HH Goa 2026, Task 2

A voice question → transcription → retrieval → grounded answer pipeline, built
as a harnessed, guardrailed FastAPI service over the
[ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) dataset.

```
 Voice Input → Speech-to-Text (Sarvam) → Input Guardrails → Vector DB Retrieval
                                                                  │
                                                                  ▼
                              Output Guardrails ◄─ Answer Generation
                                     │
                                     ▼
                              Structured Response
```

## Why this design

- **Generation: Sarvam by default**, not Anthropic. The Sarvam Chat Completion
  API is OpenAI-compatible and billed against the same universal, free-tier
  credits as STT — so one Sarvam account covers the whole live pipeline with
  no separate paid provider required to get a working demo. `AnthropicGenerator`
  is still implemented (`app/generation/generator.py`) if you'd rather use
  Claude and are fine with separate paid billing — just set `LLM_PROVIDER=anthropic`.
- **Vector DB: local FAISS**, not a hosted vector DB. Retrieval latency needs to
  be sub-200ms; a network round trip to a hosted DB makes that structurally
  impossible, so we keep an in-process flat inner-product index (cosine
  similarity over normalized embeddings).
- **Chunking is multi-strategy on purpose** — see below. This was the most
  explicit requirement in the brief ("chunking strategy should be vast").
- **A real harness**, not a prompt-in/text-out call — see "Harness" below.
- **Guardrails on both sides of generation** — input-side (unsafe, off-topic)
  and output-side (empty retrieval, hallucination/grounding).

## Chunking strategies (`app/ingestion/chunkers.py`)

| Strategy | What it does |
|---|---|
| `fixed` | Sliding word-count window with configurable overlap. Baseline. |
| `recursive` | Splits paragraph → sentence → word, packing units up to `CHUNK_SIZE` and carrying `CHUNK_OVERLAP` words of context into the next chunk. Preserves natural boundaries. |
| `semantic` | Embeds consecutive sentences, cuts a new chunk where cosine similarity between neighbors drops below a threshold (topic-shift detection) instead of at a fixed size. |
| `metadata_aware` | Wraps another chunker and enriches every chunk with dataset-native metadata (query id, language, relevance label) + derived metadata (chunk length, position-in-doc), for filtered/boosted retrieval later. |
| `hybrid` **(default)** | Recursive structural split first (cheap, fast); any chunk that's still oversized/heterogeneous gets re-split semantically; everything is metadata-enriched. Most chunks take the cheap path, the expensive embedding-based pass only runs where needed. |

Set `CHUNK_STRATEGY` in `.env` to pick one. Retrieval (`app/retrieval/retriever.py`)
additionally reranks raw ANN top-k using metadata signals (position-in-doc, chunk
length) on top of cosine similarity, rather than trusting raw vector score alone.

## The harness (`app/pipeline/orchestrator.py`)

This is not a single prompt-in/text-out call. `RagVoiceOrchestrator.run()`:

- Uses **structured Pydantic I/O** at every stage boundary (`app/schemas.py`) —
  transcription, retrieval, generation, and guardrail verdicts are all typed
  models, not raw dicts.
- **Retries** transient stage failures with backoff (`_run_stage`, and provider
  clients additionally retry their own HTTP calls via `tenacity`).
- **Recovers from errors** gracefully — a stage failure produces a structured
  `PipelineResponse(ok=False, errors=[...])`, never an unhandled exception.
- **Short-circuits** on guardrail refusals with a specific, human-readable reason
  instead of forcing a query through to generation.
- Reports a **full per-stage latency breakdown** on every response.

## Guardrails (`app/guardrails/`)

**Input side** (before spending a retrieval/generation call):
- `UnsafeInputGuardrail` — pattern-screens for unsafe requests and prompt-injection
  attempts ("ignore previous instructions", "reveal your system prompt", etc.)
- `OffTopicGuardrail` — compares the query embedding to the corpus centroid;
  queries too dissimilar from anything in the indexed corpus are refused before
  retrieval, rather than silently returning irrelevant top-k chunks.

**Output side** (after generation, before returning to the user):
- `EmptyRetrievalGuardrail` — refuses before calling generation at all if
  retrieval returned nothing above a relevance floor.
- `GroundingGuardrail` — checks generated-answer/retrieved-context word overlap
  AND the generator's self-reported citations; either signal failing routes to
  a refusal. An explicit "I don't have enough information..." response from the
  generator is treated as a valid grounded outcome, not a failure.

The system returns `refused: true` + a reason rather than an answer whenever any
of these trip — see `PipelineResponse` in `app/schemas.py`.

## ⚠️ On the 200ms latency target (read this before checking the numbers)

The task's 200ms target is realistic for **chunking + vector DB retrieval**,
which is what `scripts/benchmark_latency.py` reports as `retrieval_only`. It is
**not** realistic for the full pipeline including STT and LLM generation —
those are external network calls to Sarvam/ElevenLabs and an LLM API, and
typically add 1-5+ seconds combined regardless of how the RAG layer is built.
We report both numbers separately, honestly, rather than construct a benchmark
that hides that.

### Latency report (from an actual run of `scripts/benchmark_latency.py --n 40`)

Run in this sandbox in fully offline `PIPELINE_MODE=mock` (no network — mock STT,
mock/extractive generation, hash-fallback embeddings since `sentence-transformers`
couldn't be downloaded here). Numbers are in `results/latency_report.json`;
re-run locally with real dependencies installed for representative numbers on
the real embedding model:

| Metric | P50 | P70 | P100 (max) |
|---|---|---|---|
| **Retrieval only** (chunking already built + vector search) | 0.05 ms | 0.05 ms | 0.48 ms |
| **Full pipeline** (mock STT → guardrails → retrieval → generation → guardrails) | 0.03 ms | 0.14 ms | 0.39 ms |

`retrieval_only_under_200ms_p100: true`. In **live mode** with real Sarvam +
real LLM calls, expect `retrieval_only` to stay in the same low-ms range (it's
local FAISS regardless of mode) while `end_to_end` will be dominated by the two
external API round trips — budget and report that separately in your final
submission once you've plugged in real keys, using the same script.

## Project layout

```
app/
  stt/            Sarvam / ElevenLabs / Mock clients behind one interface
  ingestion/      dataset loader, chunking strategies, indexer
  retrieval/      embedder, FAISS vector store, retriever + reranking
  generation/     LLM generation (mock / Anthropic), prompt templates
  guardrails/     input + output guardrails
  pipeline/       the orchestrator/harness
  main.py         FastAPI app (/query, /query/text, /health, /demo)
scripts/
  build_index.py         build the FAISS index from MSMARCO-XI
  benchmark_latency.py   P50/P70/P100 latency report
static/index.html        minimal browser demo (mic recording + text fallback)
tests/                    pytest suite for chunkers + guardrails
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: set SARVAM_API_KEY / LLM_API_KEY and PIPELINE_MODE=live for real calls,
# or leave PIPELINE_MODE=mock to run fully offline with no keys.

python scripts/build_index.py --langs hi --num-per-lang 200
uvicorn app.main:app --reload
# open http://localhost:8000/demo
```

Run tests: `pytest tests/`
Run the latency benchmark: `python scripts/benchmark_latency.py --n 50`

## API

- `POST /query` — multipart form, `file` = audio blob → full voice pipeline
- `POST /query/text` — `{"text": "..."}` → same harness, skips STT (useful for
  load testing / grading without recording audio each time)
- `GET /health` — liveness + index status
- `GET /demo` — browser mic-recording demo UI

## Known limitations / what a production version would add

- Off-topic and grounding thresholds (`OFFTOPIC_MIN_SIMILARITY`,
  `GROUNDING_MIN_OVERLAP` in `.env`) are heuristics tuned by hand, not learned —
  a production system would calibrate these against a labeled eval set.
- The unsafe-input guardrail is a regex screen, not a trained classifier;
  swap in a hosted moderation endpoint for production traffic.
- `MockGenerator` is extractive (returns literal retrieved text) so the
  pipeline is fully testable without any API key — set `PIPELINE_MODE=live`
  and configure `LLM_API_KEY` for real generative answers.
