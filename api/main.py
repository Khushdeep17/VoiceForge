import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routes.speak import router as speak_router
from api.routes.health import router as health_router

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(
    title="VoiceForge",
    description="Emotion-aware text-to-speech with voice personas",
    version="2.0.0",
)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

app.include_router(speak_router)
app.include_router(health_router)
