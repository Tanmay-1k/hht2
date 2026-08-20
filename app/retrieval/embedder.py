"""
Thin wrapper around sentence-transformers so both the semantic chunker and
the vector retriever share one loaded model (loaded lazily, once).
Falls back to a deterministic hashing-based pseudo-embedding if the model
can't be downloaded (offline sandbox), so the pipeline still runs and
latency benchmarks still mean something structurally, even if not
semantically state-of-the-art.
"""
from __future__ import annotations
import hashlib
import numpy as np

from app.config import settings

_model = None
_dim = 384  # matches all-MiniLM-L6-v2


def _load_model():
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(settings.embedding_model)
    except Exception as e:
        print(f"[embedder] Falling back to hashing embedder — could not load {settings.embedding_model}: {e}")
        _model = "hash_fallback"
    return _model


def _hash_embed(texts: list[str]) -> np.ndarray:
    """Deterministic bag-of-hashed-tokens embedding, unit-normalized.
    Not semantically strong, but keeps the system fully functional offline."""
    vecs = np.zeros((len(texts), _dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % _dim
            vecs[i, h] += 1.0
        norm = np.linalg.norm(vecs[i])
        if norm > 0:
            vecs[i] /= norm
    return vecs


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, _dim), dtype=np.float32)
    model = _load_model()
    if model == "hash_fallback":
        return _hash_embed(texts)
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)


def embed_dim() -> int:
    return _dim
