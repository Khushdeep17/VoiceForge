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

# mount outputs folder
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

templates = Jinja2Templates(directory="templates")


class TextInput(BaseModel):
    text: str


# API endpoint
@app.post("/speak")
def speak(data: TextInput):

    text = data.text

    emotion, intensity = detect_emotion(text)

    rate, volume, style = get_voice_params(emotion, intensity)

    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)

    generate_audio(text, rate, volume, filepath)

    return {
        "emotion": emotion,
        "intensity": round(intensity, 3),
        "voice_style": style,
        "audio_file": filepath
    }


# UI endpoint
@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )


@app.post("/speak-ui", response_class=HTMLResponse)
def speak_ui(request: Request, text: str = Form(...)):

    emotion, intensity = detect_emotion(text)

    rate, volume, style = get_voice_params(emotion, intensity)

    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)

    generate_audio(text, rate, volume, filepath)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "emotion": emotion,
            "style": style,
            "audio_file": filepath
        }
    )