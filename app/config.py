"""
Central configuration. Everything is env-driven so the same code runs in
`mock` mode (no keys, no network — used for local dev / grading without
credentials) and `live` mode (real Sarvam/ElevenLabs + LLM calls).
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    pipeline_mode: str = Field(default="mock")  # mock | live

    stt_provider: str = Field(default="sarvam")  # sarvam | elevenlabs | mock
    sarvam_api_key: str | None = None
    elevenlabs_api_key: str | None = None

    llm_provider: str = Field(default="mock")  # sarvam | anthropic | openai | mock
    llm_api_key: str | None = None
    llm_model: str = "sarvam-105b"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_index_path: str = "./data/faiss.index"
    vector_metadata_path: str = "./data/metadata.jsonl"

    chunk_strategy: str = "hybrid"  # fixed | recursive | semantic | metadata_aware | hybrid
    chunk_size: int = 256
    chunk_overlap: int = 48

    grounding_min_overlap: float = 0.35
    offtopic_min_similarity: float = 0.28

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
