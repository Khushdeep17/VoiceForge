# api/main.py

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Routers
from api.routes.speak import router as speak_router
from api.routes.health import router as health_router


# ---------------------------------------------------
# Directory setup
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------

app = FastAPI(
    title="VoiceForge",
    description="Emotion-aware text-to-speech with voice personas",
    version="2.0.0",
)


# ---------------------------------------------------
# Static Files
# ---------------------------------------------------

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs"
)


# ---------------------------------------------------
# Routes
# ---------------------------------------------------

app.include_router(speak_router)
app.include_router(health_router)


# ---------------------------------------------------
# Root endpoint
# ---------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "VoiceForge",
        "status": "running",
        "version": "2.0.0"
    }