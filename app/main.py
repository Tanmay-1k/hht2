"""
FastAPI entrypoint.

POST /query          - multipart audio upload -> full voice RAG pipeline
POST /query/text      - JSON {"text": "..."} -> skips STT, exercises the
                         retrieval+generation+guardrail harness directly
                         (handy for load-testing and demoing without mic audio)
GET  /health          - liveness + index status
"""
from __future__ import annotations
import os

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import settings
from app.guardrails.input_guardrails import compute_corpus_centroid
from app.pipeline.orchestrator import RagVoiceOrchestrator
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore
from app.schemas import PipelineResponse
from app.stt.factory import get_stt_client

app = FastAPI(title="Voice-Enabled RAG — HH Goa 2026", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.mount("/demo", StaticFiles(directory="static", html=True), name="demo")

_state: dict = {"orchestrator": None, "store": None}


@app.on_event("startup")
def startup() -> None:
    store = VectorStore(settings.vector_index_path, settings.vector_metadata_path)
    if os.path.exists(settings.vector_index_path) and os.path.exists(settings.vector_metadata_path):
        store.load()
    else:
        # Build a small index on the fly if none exists yet (e.g. first run in a
        # fresh clone) so `/query` never 500s just because build_index.py wasn't
        # run manually first.
        from app.ingestion.indexer import build_index
        store = build_index()

    centroid = compute_corpus_centroid(store)
    retriever = Retriever(store, top_k=5)
    stt = get_stt_client()
    _state["orchestrator"] = RagVoiceOrchestrator(stt, retriever, corpus_centroid=centroid)
    _state["store"] = store


@app.get("/health")
def health():
    store = _state.get("store")
    return {
        "status": "ok",
        "pipeline_mode": settings.pipeline_mode,
        "stt_provider": settings.stt_provider,
        "chunk_strategy": settings.chunk_strategy,
        "indexed_chunks": store.index.ntotal if store and store.index else 0,
    }


class TextQuery(BaseModel):
    text: str
    top_k: int = 5


@app.post("/query/text", response_model=PipelineResponse)
def query_text(payload: TextQuery):
    orchestrator: RagVoiceOrchestrator = _state["orchestrator"]
    if orchestrator is None:
        raise HTTPException(503, "Pipeline not initialized yet.")
    # route text straight through as if it were "already transcribed" audio
    # bytes, so the exact same harness code path (incl. guardrails) runs.
    response = orchestrator.run(payload.text.encode("utf-8"), top_k=payload.top_k)
    return response


@app.post("/query", response_model=PipelineResponse)
async def query_audio(file: UploadFile = File(...), language_hint: str | None = Form(None), top_k: int = Form(5)):
    orchestrator: RagVoiceOrchestrator = _state["orchestrator"]
    if orchestrator is None:
        raise HTTPException(503, "Pipeline not initialized yet.")
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio file.")
    response = orchestrator.run(audio_bytes, language_hint=language_hint, top_k=top_k)
    return response
