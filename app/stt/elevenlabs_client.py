"""
ElevenLabs Speech-to-Text client (kept as a drop-in alternative to Sarvam —
switch STT_PROVIDER=elevenlabs in .env to use this instead).

Docs: https://elevenlabs.io/docs/api-reference/speech-to-text
Endpoint used: POST https://api.elevenlabs.io/v1/speech-to-text
"""
import time
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings
from app.schemas import TranscriptionResult
from app.stt.base import BaseSTT

ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"


class ElevenLabsSTT(BaseSTT):
    provider_name = "elevenlabs"

    def __init__(self, api_key: str | None = None, timeout_s: float = 15.0):
        self.api_key = api_key or settings.elevenlabs_api_key
        self.timeout_s = timeout_s
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY is not set. Set it in .env or pass explicitly.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.3, min=0.3, max=3),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        reraise=True,
    )
    def _call_api(self, audio_bytes: bytes) -> dict:
        headers = {"xi-api-key": self.api_key}
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        data = {"model_id": "scribe_v1"}
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(ELEVENLABS_STT_URL, headers=headers, files=files, data=data)
            resp.raise_for_status()
            return resp.json()

    def transcribe(self, audio_bytes: bytes, language_hint: str | None = None) -> TranscriptionResult:
        start = time.perf_counter()
        try:
            payload = self._call_api(audio_bytes)
        except Exception as e:
            raise RuntimeError(f"ElevenLabs STT failed after retries: {e}") from e
        elapsed_ms = (time.perf_counter() - start) * 1000

        return TranscriptionResult(
            text=payload.get("text", ""),
            language=payload.get("language_code", language_hint),
            confidence=None,
            provider=self.provider_name,
            raw_latency_ms=elapsed_ms,
        )
