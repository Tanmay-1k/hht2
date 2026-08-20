from abc import ABC, abstractmethod
from app.schemas import TranscriptionResult


class BaseSTT(ABC):
    """Common interface so Sarvam / ElevenLabs / Mock are interchangeable."""

    provider_name: str = "base"

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, language_hint: str | None = None) -> TranscriptionResult:
        ...
