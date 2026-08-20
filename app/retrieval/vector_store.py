"""
Local FAISS vector store (flat inner-product index over normalized vectors
== cosine similarity), with a JSONL metadata sidecar keyed by row index.
Chosen over a hosted vector DB so retrieval latency isn't dominated by a
network round trip — this is what makes the <200ms retrieval-only target
realistic (see scripts/benchmark_latency.py).
"""
from __future__ import annotations
import json
import os
from dataclasses import asdict

import faiss
import numpy as np

from app.ingestion.chunkers import Chunk
from app.retrieval.embedder import embed_texts, embed_dim


class VectorStore:
    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index: faiss.Index | None = None
        self.metadata: list[dict] = []

    def build(self, chunks: list[Chunk]) -> None:
        texts = [c.text for c in chunks]
        vectors = embed_texts(texts)
        dim = vectors.shape[1] if len(vectors) else embed_dim()
        self.index = faiss.IndexFlatIP(dim)
        if len(vectors):
            self.index.add(vectors)
        self.metadata = [asdict(c) for c in chunks]

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            for row in self.metadata:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def load(self) -> None:
        self.index = faiss.read_index(self.index_path)
        self.metadata = []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line))

    def search(self, query: str, k: int = 5, metadata_filter=None) -> list[tuple[dict, float]]:
        if self.index is None:
            raise RuntimeError("Vector store not built/loaded yet.")
        query_vec = embed_texts([query])
        # over-fetch when a metadata filter is supplied, since some hits will be dropped
        fetch_k = k * 5 if metadata_filter else k
        fetch_k = min(fetch_k, self.index.ntotal) or 1
        scores, idxs = self.index.search(query_vec, fetch_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            if metadata_filter and not metadata_filter(meta):
                continue
            results.append((meta, float(score)))
            if len(results) >= k:
                break
        return results
