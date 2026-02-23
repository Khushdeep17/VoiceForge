from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import uuid
import os

from emotion import detect_emotion
from tts_engine import get_voice_params, generate_audio


app = FastAPI(title="Empathy Engine")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

templates = Jinja2Templates(directory="templates")


class TextInput(BaseModel):
    text: str


# ---------------- API Endpoint ----------------
@app.post("/speak")
def speak(data: TextInput):

    text = data.text

    emotion, confidence, viz_file = detect_emotion(text, mode="hybrid")

    rate, volume, style = get_voice_params(emotion, confidence)

    audio_filename = f"{uuid.uuid4()}.wav"
    audio_filepath = os.path.join(OUTPUT_DIR, audio_filename)

    generate_audio(
        text=text,
        rate=rate,
        volume=volume,
        filename=audio_filepath
    )

    return {
        "emotion": emotion,
        "confidence": round(confidence, 3),
        "voice_parameters": {
            "rate": rate,
            "volume": volume
        },
        "voice_style": style,
        "audio_file": f"/outputs/{audio_filename}",
        "emotion_visualization": viz_file
    }


# ---------------- Web UI ----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/speak-ui", response_class=HTMLResponse)
def speak_ui(request: Request, text: str = Form(...)):

    emotion, confidence, viz_file = detect_emotion(text, mode="hybrid")

    rate, volume, style = get_voice_params(emotion, confidence)

    audio_filename = f"{uuid.uuid4()}.wav"
    audio_filepath = os.path.join(OUTPUT_DIR, audio_filename)

    generate_audio(
        text=text,
        rate=rate,
        volume=volume,
        filename=audio_filepath
    )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "emotion": emotion,
            "confidence": round(confidence, 3),
            "style": style,
            "rate": rate,
            "volume": volume,
            "audio_file": f"/outputs/{audio_filename}",
            "viz_file": viz_file
        }
    )