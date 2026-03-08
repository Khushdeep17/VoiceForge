import os
import uuid

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from voiceforge.emotion.detector import detect_emotion
from voiceforge.tts.engine import get_voice_params, generate_audio
from voiceforge.tts.personas import get_persona
from api.schemas import SpeakRequest, SpeakResponse, VoiceParameters


router = APIRouter()
templates = Jinja2Templates(directory="templates")
OUTPUT_DIR = "outputs"


@router.post("/speak", response_model=SpeakResponse)
def speak(data: SpeakRequest):
    """JSON API endpoint. Supports optional persona override."""
    result = detect_emotion(data.text, mode=data.mode)


    persona = get_persona(data.persona) if data.persona else None
    if persona:
        rate = persona.rate
        volume = persona.volume
        style = persona.style
    else:
        rate, volume, style = get_voice_params(result.emotion, result.confidence)



    audio_filename = f"{uuid.uuid4()}.wav"
    audio_filepath = os.path.join(OUTPUT_DIR, audio_filename)
    generate_audio(text=data.text, rate=rate, volume=volume, filename=audio_filepath)

    return SpeakResponse(
        emotion=result.emotion,
        confidence=round(result.confidence, 3),
        voice_parameters=VoiceParameters(rate=rate, volume=volume),
        voice_style=style,
        audio_file=f"/outputs/{audio_filename}",
        emotion_visualization=result.viz_path,
    )


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/speak-ui", response_class=HTMLResponse)
def speak_ui(
    request: Request,
    text: str = Form(...),
    persona: str = Form(None)
):
    """Form-based UI endpoint."""
    result = detect_emotion(text, mode="hybrid")
    rate, volume, style = get_voice_params(result.emotion, result.confidence)

    audio_filename = f"{uuid.uuid4()}.wav"
    audio_filepath = os.path.join(OUTPUT_DIR, audio_filename)
    generate_audio(text=text, rate=rate, volume=volume, filename=audio_filepath)

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
