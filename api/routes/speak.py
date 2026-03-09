import os
import time
import uuid

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from voiceforge.emotion.detector import detect_emotion
from voiceforge.tts.engine import get_voice_params, generate_audio
from voiceforge.tts.personas import get_persona
from voiceforge.metrics.tracker import log_inference
from voiceforge.storage.s3 import upload_and_get_url
from api.schemas import SpeakRequest, SpeakResponse, VoiceParameters


router = APIRouter()
templates = Jinja2Templates(directory="templates")
OUTPUT_DIR = "outputs"


@router.post("/speak", response_model=SpeakResponse)
def speak(data: SpeakRequest):
    t_start = time.perf_counter()

    # emotion detection — timed separately so we can log it
    t_emotion = time.perf_counter()
    result = detect_emotion(data.text, mode=data.mode)
    emotion_latency_ms = (time.perf_counter() - t_emotion) * 1000

    # persona overrides emotion-derived voice params
    persona = get_persona(data.persona) if data.persona else None
    if persona:
        rate, volume, style = persona.rate, persona.volume, persona.style
    else:
        rate, volume, style = get_voice_params(result.emotion, result.confidence)

    audio_filename = f"{uuid.uuid4()}.wav"
    audio_filepath = os.path.join(OUTPUT_DIR, audio_filename)
    generate_audio(text=data.text, rate=rate, volume=volume, filename=audio_filepath)

    # S3 upload — non-blocking, local fallback if it fails
    audio_s3 = upload_and_get_url(audio_filepath)
    if result.viz_path:
        upload_and_get_url(result.viz_path)

    total_latency_ms = (time.perf_counter() - t_start) * 1000

    log_inference(
        text=data.text,
        emotion=result.emotion,
        confidence=result.confidence,
        scores=result.scores,
        rate=rate,
        volume=volume,
        style=style,
        persona=data.persona,
        mode=data.mode,
        emotion_latency_ms=emotion_latency_ms,
        total_latency_ms=total_latency_ms,
    )

    return SpeakResponse(
        emotion=result.emotion,
        confidence=round(result.confidence, 3),
        voice_parameters=VoiceParameters(rate=rate, volume=volume),
        voice_style=style,
        audio_file=f"/outputs/{audio_filename}",
        s3_audio_url=audio_s3.get("presigned_url"),
        emotion_visualization=result.viz_path,
    )


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/speak-ui", response_class=HTMLResponse)
def speak_ui(request: Request, text: str = Form(...)):
    t_start = time.perf_counter()

    t_emotion = time.perf_counter()
    result = detect_emotion(text, mode="hybrid")
    emotion_latency_ms = (time.perf_counter() - t_emotion) * 1000

    rate, volume, style = get_voice_params(result.emotion, result.confidence)

    audio_filename = f"{uuid.uuid4()}.wav"
    audio_filepath = os.path.join(OUTPUT_DIR, audio_filename)
    generate_audio(text=text, rate=rate, volume=volume, filename=audio_filepath)

    # S3 upload for UI requests too
    upload_and_get_url(audio_filepath)
    if result.viz_path:
        upload_and_get_url(result.viz_path)

    total_latency_ms = (time.perf_counter() - t_start) * 1000

    log_inference(
        text=text,
        emotion=result.emotion,
        confidence=result.confidence,
        scores=result.scores,
        rate=rate,
        volume=volume,
        style=style,
        persona=None,
        mode="hybrid",
        emotion_latency_ms=emotion_latency_ms,
        total_latency_ms=total_latency_ms,
    )

    return templates.TemplateResponse("index.html", {
        "request": request,
        "emotion": result.emotion,
        "confidence": round(result.confidence, 3),
        "style": style,
        "rate": rate,
        "volume": volume,
        "audio_file": f"/outputs/{audio_filename}",
        "viz_file": result.viz_path,
    })
