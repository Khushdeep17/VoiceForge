from typing import Optional
from pydantic import BaseModel


class SpeakRequest(BaseModel):
    text: str
    mode: str = "hybrid"
    persona: Optional[str] = None


class VoiceParameters(BaseModel):
    rate: int
    volume: float


class SpeakResponse(BaseModel):
    emotion: str
    confidence: float
    voice_parameters: VoiceParameters
    voice_style: str
    audio_file: str                          # local path — always present
    s3_audio_url: Optional[str] = None      # presigned S3 URL — present when S3 configured
    emotion_visualization: Optional[str] = None
