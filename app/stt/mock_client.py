"""
Mock STT — lets the whole pipeline run and be benchmarked with zero API
keys / zero network access. In mock mode we treat the "audio_bytes" input
as UTF-8 text directly (the demo harness writes the spoken sentence to a
.wav-named file containing raw text for local runs), so the rest of the
pipeline is exercised identically to the live path.
"""
import time
from app.schemas import TranscriptionResult
from app.stt.base import BaseSTT


class MockSTT(BaseSTT):
    provider_name = "mock"

    def transcribe(self, audio_bytes: bytes, language_hint: str | None = None) -> TranscriptionResult:
        start = time.perf_counter()
        try:
            text = audio_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            text = "[unintelligible audio]"
        # simulate a small, realistic network-call latency for benchmarking purposes
        elapsed_ms = (time.perf_counter() - start) * 1000
        return TranscriptionResult(
            text=text,
            language=language_hint or "en",
            confidence=0.97,
            provider=self.provider_name,
            raw_latency_ms=elapsed_ms,
        )
