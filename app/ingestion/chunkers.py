"""
Chunking strategies.

We deliberately implement several distinct strategies rather than one
fixed-size splitter, because MSMARCO-XI passages vary a lot in length and
structure (short factoid answers vs. long explanatory passages), and
retrieval quality is very sensitive to chunk boundaries:

1. FixedSizeChunker      - token/word-count windows with configurable overlap.
                            Baseline, cheap, predictable chunk sizes.
2. RecursiveChunker       - splits on a hierarchy of separators
                            (paragraph -> sentence -> word) and only falls
                            back to a harder split when a unit is still too
                            big. Preserves natural language boundaries.
3. SemanticChunker        - embeds consecutive sentences and cuts a new
                            chunk boundary where cosine similarity between
                            neighboring sentences drops below a threshold
                            (topic shift detection), rather than a fixed size.
4. MetadataAwareChunker   - wraps another chunker but attaches and can key
                            off dataset-native metadata (query_id, passage
                            id, doc position, language) so retrieval can
                            filter/boost using this metadata later.
5. HybridChunker          - runs recursive chunking as the primary
                            structural pass, then applies semantic
                            re-splitting on any chunk that is still long and
                            topically heterogeneous. This is the default
                            ("hybrid") strategy and is what
                            CHUNK_STRATEGY=hybrid selects.

All strategies return a common Chunk dataclass so the indexer/retriever
don't need to know which strategy produced a given chunk (though the
strategy name is stored in metadata for analysis/debugging).
"""
from __future__ import annotations
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_doc_id: str
    strategy: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _word_tokenize(text: str) -> list[str]:
    return text.split()


def _sentence_split(text: str) -> list[str]:
    # Lightweight sentence splitter (avoids an nltk punkt download dependency
    # at index-build time). Good enough for MS MARCO-style passages.
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?।])\s+", text)  # '।' = Devanagari danda, for Indic text
    return [s.strip() for s in sentences if s.strip()]


class BaseChunker:
    name = "base"

    def split(self, text: str, doc_id: str, extra_metadata: dict[str, Any] | None = None) -> list[Chunk]:
        raise NotImplementedError


class FixedSizeChunker(BaseChunker):
    """Sliding window over words with overlap. Simple, robust baseline."""
    name = "fixed"

    def __init__(self, chunk_size: int = 256, overlap: int = 48):
        assert overlap < chunk_size, "overlap must be smaller than chunk_size"
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str, doc_id: str, extra_metadata: dict[str, Any] | None = None) -> list[Chunk]:
        words = _word_tokenize(text)
        if not words:
            return []
        chunks = []
        step = self.chunk_size - self.overlap
        pos = 0
        idx = 0
        while pos < len(words):
            window = words[pos: pos + self.chunk_size]
            chunk_text = " ".join(window)
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                text=chunk_text,
                source_doc_id=doc_id,
                strategy=self.name,
                position=idx,
                metadata={**(extra_metadata or {}), "word_start": pos, "word_end": pos + len(window)},
            ))
            pos += step
            idx += 1
        return chunks


class RecursiveChunker(BaseChunker):
    """
    Splits on a separator hierarchy (paragraphs -> sentences -> words),
    trying the coarsest separator first and only recursing into a finer
    one when a segment still exceeds chunk_size. This keeps chunks aligned
    to natural language boundaries far more often than fixed windows.
    """
    name = "recursive"

    def __init__(self, chunk_size: int = 256, overlap: int = 48):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._fallback = FixedSizeChunker(chunk_size, overlap)

    def _merge_units(self, units: list[str]) -> list[str]:
        """Greedily pack small units (sentences/paragraphs) into chunks near chunk_size words,
        carrying `overlap` words of context from the tail of the previous chunk."""
        merged, current, current_len = [], [], 0
        for unit in units:
            unit_len = len(_word_tokenize(unit))
            if current_len + unit_len > self.chunk_size and current:
                merged.append(" ".join(current))
                # start next chunk with overlap words carried over
                overlap_words = _word_tokenize(merged[-1])[-self.overlap:]
                current = [" ".join(overlap_words)] if overlap_words else []
                current_len = len(overlap_words)
            current.append(unit)
            current_len += unit_len
        if current:
            merged.append(" ".join(current))
        return merged

    def split(self, text: str, doc_id: str, extra_metadata: dict[str, Any] | None = None) -> list[Chunk]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()] or [text]
        units: list[str] = []
        for para in paragraphs:
            if len(_word_tokenize(para)) <= self.chunk_size:
                units.append(para)
            else:
                units.extend(_sentence_split(para))

        merged_chunks = self._merge_units(units)
        result = []
        for idx, chunk_text in enumerate(merged_chunks):
            if len(_word_tokenize(chunk_text)) > self.chunk_size * 1.5:
                # still too big (e.g. one giant run-on sentence) -> hard fallback split
                for sub in self._fallback.split(chunk_text, doc_id, extra_metadata):
                    sub.strategy = f"{self.name}+fixed_fallback"
                    sub.position = idx
                    result.append(sub)
            else:
                result.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=chunk_text,
                    source_doc_id=doc_id,
                    strategy=self.name,
                    position=idx,
                    metadata=extra_metadata or {},
                ))
        return result


class SemanticChunker(BaseChunker):
    """
    Embeds sentences and cuts a new chunk boundary at topic-shift points
    (where cosine similarity to the previous sentence drops below
    `similarity_threshold`), instead of at a fixed word count. Requires an
    embedding function to be injected (shared with the retriever's embedder
    so we don't load the model twice).
    """
    name = "semantic"

    def __init__(self, embed_fn, similarity_threshold: float = 0.55, max_chunk_size: int = 400):
        self.embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size

    @staticmethod
    def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0

    def split(self, text: str, doc_id: str, extra_metadata: dict[str, Any] | None = None) -> list[Chunk]:
        sentences = _sentence_split(text)
        if len(sentences) <= 1:
            if not sentences:
                return []
            return [Chunk(str(uuid.uuid4()), sentences[0], doc_id, self.name, 0, extra_metadata or {})]

        embeddings = self.embed_fn(sentences)
        chunks, current, current_len = [], [sentences[0]], len(_word_tokenize(sentences[0]))

        for i in range(1, len(sentences)):
            sim = self._cos_sim(embeddings[i - 1], embeddings[i])
            sent_len = len(_word_tokenize(sentences[i]))
            topic_shift = sim < self.similarity_threshold
            too_long = current_len + sent_len > self.max_chunk_size
            if topic_shift or too_long:
                chunks.append(" ".join(current))
                current, current_len = [sentences[i]], sent_len
            else:
                current.append(sentences[i])
                current_len += sent_len
        if current:
            chunks.append(" ".join(current))

        return [
            Chunk(str(uuid.uuid4()), c, doc_id, self.name, idx, extra_metadata or {})
            for idx, c in enumerate(chunks)
        ]


class MetadataAwareChunker(BaseChunker):
    """
    Wraps a base chunker and enriches every resulting chunk with dataset
    metadata (e.g. MSMARCO query_id, passage rank/relevance label,
    language) plus lightweight derived metadata (chunk length, position
    ratio within doc). This metadata is what lets the retriever later do
    filtered/boosted search (e.g. "only relevant-labeled passages",
    "prefer chunks near the start of a doc") instead of pure vector
    similarity.
    """
    name = "metadata_aware"

    def __init__(self, base_chunker: BaseChunker):
        self.base_chunker = base_chunker

    def split(self, text: str, doc_id: str, extra_metadata: dict[str, Any] | None = None) -> list[Chunk]:
        base_chunks = self.base_chunker.split(text, doc_id, extra_metadata)
        n = len(base_chunks) or 1
        for i, c in enumerate(base_chunks):
            c.strategy = f"{self.name}({self.base_chunker.name})"
            c.metadata.update({
                "n_words": len(_word_tokenize(c.text)),
                "position_ratio": round(i / n, 3),
                "is_doc_start": i == 0,
                "is_doc_end": i == n - 1,
            })
        return base_chunks


class HybridChunker(BaseChunker):
    """
    Default strategy. Recursive structural split first (fast, boundary-
    aware), then semantic re-splitting is applied only to chunks that come
    out oversized/heterogeneous, then the whole thing is wrapped with
    metadata enrichment. This gives most chunks the cheap recursive path
    and reserves the more expensive embedding-based semantic pass for the
    minority of chunks that actually need it.
    """
    name = "hybrid"

    def __init__(self, embed_fn, chunk_size: int = 256, overlap: int = 48):
        self.recursive = RecursiveChunker(chunk_size, overlap)
        self.semantic = SemanticChunker(embed_fn, max_chunk_size=int(chunk_size * 1.3))
        self.metadata_wrapper = MetadataAwareChunker(self.recursive)
        self.chunk_size = chunk_size

    def split(self, text: str, doc_id: str, extra_metadata: dict[str, Any] | None = None) -> list[Chunk]:
        recursive_chunks = self.recursive.split(text, doc_id, extra_metadata)
        final: list[Chunk] = []
        for c in recursive_chunks:
            n_words = len(_word_tokenize(c.text))
            if n_words > self.chunk_size * 1.4:
                # re-split this specific oversized/heterogeneous chunk semantically
                sub_chunks = self.semantic.split(c.text, doc_id, extra_metadata)
                for sc in sub_chunks:
                    sc.strategy = f"{self.name}(recursive->semantic)"
                final.extend(sub_chunks)
            else:
                final.append(c)

        n = len(final) or 1
        for i, c in enumerate(final):
            c.metadata.update({
                "n_words": len(_word_tokenize(c.text)),
                "position_ratio": round(i / n, 3),
                "is_doc_start": i == 0,
                "is_doc_end": i == n - 1,
            })
        return final


def get_chunker(strategy: str, embed_fn=None, chunk_size: int = 256, overlap: int = 48) -> BaseChunker:
    strategy = strategy.lower()
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size, overlap)
    if strategy == "recursive":
        return RecursiveChunker(chunk_size, overlap)
    if strategy == "semantic":
        if embed_fn is None:
            raise ValueError("semantic chunking requires an embed_fn")
        return SemanticChunker(embed_fn, max_chunk_size=int(chunk_size * 1.3))
    if strategy == "metadata_aware":
        return MetadataAwareChunker(RecursiveChunker(chunk_size, overlap))
    if strategy == "hybrid":
        if embed_fn is None:
            raise ValueError("hybrid chunking requires an embed_fn")
        return HybridChunker(embed_fn, chunk_size, overlap)
    raise ValueError(f"Unknown chunk strategy: {strategy}")
