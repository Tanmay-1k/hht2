"""
Runs a batch of test queries through the pipeline and reports P50/P70/P100
latency, broken out per stage, plus two headline numbers:

  - "retrieval_only"  : chunking-already-done + vector DB search only.
                        This is the number the <200ms target is realistic
                        for, and is reported separately as required.
  - "end_to_end"       : full STT -> guardrails -> retrieval -> generation
                        -> guardrails, for honesty about real-world latency
                        (dominated by external API calls in live mode).

Usage:
    python scripts/benchmark_latency.py --n 50
    python scripts/benchmark_latency.py --n 50 --out results/latency_report.json
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings  # noqa: E402
from app.guardrails.input_guardrails import compute_corpus_centroid  # noqa: E402
from app.ingestion.indexer import build_index  # noqa: E402
from app.pipeline.orchestrator import RagVoiceOrchestrator  # noqa: E402
from app.retrieval.retriever import Retriever  # noqa: E402
from app.retrieval.vector_store import VectorStore  # noqa: E402
from app.stt.factory import get_stt_client  # noqa: E402

TEST_QUERIES = [
    "Who built the Taj Mahal and why?",
    "What does the RBI do?",
    "What is photosynthesis?",
    "When was the IPL founded?",
    "How tall is Mount Everest?",
    "What is the capital of France?",  # expected off-topic / not-grounded on this corpus
    "Tell me about the history of the Reserve Bank of India.",
    "Explain how plants convert sunlight into energy.",
    "Which mountain range is Mount Everest part of?",
    "What UNESCO status does the Taj Mahal have?",
]


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    data = sorted(data)
    k = (len(data) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(data) - 1)
    if f == c:
        return data[f]
    return data[f] + (data[c] - data[f]) * (k - f)


def summarize(latencies: list[float]) -> dict:
    return {
        "p50_ms": round(percentile(latencies, 50), 3),
        "p70_ms": round(percentile(latencies, 70), 3),
        "p100_ms": round(percentile(latencies, 100), 3),  # i.e. max
        "mean_ms": round(statistics.mean(latencies), 3) if latencies else 0,
        "n": len(latencies),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="number of query runs (cycles through TEST_QUERIES)")
    parser.add_argument("--out", type=str, default="results/latency_report.json")
    args = parser.parse_args()

    store = VectorStore(settings.vector_index_path, settings.vector_metadata_path)
    if os.path.exists(settings.vector_index_path):
        store.load()
    else:
        store = build_index()

    retriever = Retriever(store, top_k=5)
    stt = get_stt_client()
    centroid = compute_corpus_centroid(store)
    orchestrator = RagVoiceOrchestrator(stt, retriever, corpus_centroid=centroid)

    retrieval_only_ms, end_to_end_ms = [], []
    stage_latencies: dict[str, list[float]] = {}
    refusal_count = 0

    for i in range(args.n):
        query = TEST_QUERIES[i % len(TEST_QUERIES)]

        # retrieval-only timing (isolated, as required by the task spec)
        t0 = time.perf_counter()
        retriever.retrieve(query)
        retrieval_only_ms.append((time.perf_counter() - t0) * 1000)

        # full pipeline timing, via the real harness, text-in mode (STT mocked to
        # remove network variance from this specific benchmark; STT's own latency
        # is reported separately from live calls in the README)
        response = orchestrator.run(query.encode("utf-8"))
        end_to_end_ms.append(response.total_latency_ms)
        if response.refused:
            refusal_count += 1
        for stage, ms in response.latency_breakdown_ms.items():
            stage_latencies.setdefault(stage, []).append(ms)

    report = {
        "config": {
            "chunk_strategy": settings.chunk_strategy,
            "pipeline_mode": settings.pipeline_mode,
            "stt_provider": settings.stt_provider,
            "n_queries": args.n,
        },
        "retrieval_only": summarize(retrieval_only_ms),
        "end_to_end": summarize(end_to_end_ms),
        "per_stage": {stage: summarize(vals) for stage, vals in stage_latencies.items()},
        "refusals": refusal_count,
        "target_check": {
            "retrieval_only_under_200ms_p100": percentile(retrieval_only_ms, 100) < 200,
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved full report to {args.out}")


if __name__ == "__main__":
    main()
