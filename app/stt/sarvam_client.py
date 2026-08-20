"""
Sarvam AI Speech-to-Text client.

Docs: https://docs.sarvam.ai/api-reference-docs/speech-to-text
Endpoint used: POST https://api.sarvam.ai/speech-to-text  (multipart/form-data)

We picked Sarvam over ElevenLabs because it has native, well-tuned support
for Indian languages (Hindi + many regional languages), which matches the
MSMARCO-XI dataset (an Indic-language MS MARCO variant from AI4Bharat).
"""
import time
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings
from app.schemas import TranscriptionResult
from app.stt.base import BaseSTT

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


class SarvamSTT(BaseSTT):
    provider_name = "sarvam"

    def __init__(self, api_key: str | None = None, timeout_s: float = 15.0):
        self.api_key = api_key or settings.sarvam_api_key
        self.timeout_s = timeout_s
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY is not set. Set it in .env or pass explicitly.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.3, min=0.3, max=3),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        reraise=True,
    )
    def _call_api(self, audio_bytes: bytes, language_hint: str | None) -> dict:
        headers = {"api-subscription-key": self.api_key}
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        data = {
    "model": "saaras:v3",
    "mode": "transcribe",
}
        if language_hint:
            data["language_code"] = language_hint
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(SARVAM_STT_URL, headers=headers, files=files, data=data)
            resp.raise_for_status()
            return resp.json()

    def transcribe(self, audio_bytes: bytes, language_hint: str | None = None) -> TranscriptionResult:
        start = time.perf_counter()
        try:
            payload = self._call_api(audio_bytes, language_hint)
        except Exception as e:
            raise RuntimeError(f"Sarvam STT failed after retries: {e}") from e
        elapsed_ms = (time.perf_counter() - start) * 1000

        return TranscriptionResult(
            text=payload.get("transcript", ""),
            language=payload.get("language_code", language_hint),
            confidence=payload.get("confidence"),
            provider=self.provider_name,
            raw_latency_ms=elapsed_ms,
        )
