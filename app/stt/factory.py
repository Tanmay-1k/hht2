from app.config import settings
from app.stt.base import BaseSTT


def get_stt_client() -> BaseSTT:
    if settings.pipeline_mode == "mock" or settings.stt_provider == "mock":
        from app.stt.mock_client import MockSTT
        return MockSTT()
    if settings.stt_provider == "sarvam":
        from app.stt.sarvam_client import SarvamSTT
        return SarvamSTT()
    if settings.stt_provider == "elevenlabs":
        from app.stt.elevenlabs_client import ElevenLabsSTT
        return ElevenLabsSTT()
    raise ValueError(f"Unknown STT_PROVIDER: {settings.stt_provider}")
