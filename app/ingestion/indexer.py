from __future__ import annotations
from app.config import settings
from app.ingestion.chunkers import get_chunker
from app.ingestion.loader import load_msmarco_xi
from app.retrieval.embedder import embed_texts
from app.retrieval.vector_store import VectorStore


def build_index(languages: list[str] | None = None, num_docs_per_lang: int = 200) -> VectorStore:
    docs = load_msmarco_xi(languages=languages, num_docs_per_lang=num_docs_per_lang)
    print(f"[indexer] Loaded {len(docs)} source documents.")

    def embed_fn(texts: list[str]):
        return embed_texts(texts)

    chunker = get_chunker(
        settings.chunk_strategy,
        embed_fn=embed_fn,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunker.split(doc.text, doc.doc_id, extra_metadata=doc.metadata))
    print(f"[indexer] Produced {len(all_chunks)} chunks using strategy='{settings.chunk_strategy}'.")

    store = VectorStore(settings.vector_index_path, settings.vector_metadata_path)
    store.build(all_chunks)
    store.save()
    print(f"[indexer] Saved index to {settings.vector_index_path}")
    return store
