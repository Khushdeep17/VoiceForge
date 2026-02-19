from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import numpy as np
import io
from scipy.io.wavfile import write
import asyncio
from concurrent.futures import ThreadPoolExecutor

from model_loader import synthesize, load_model


# =====================================================
# FastAPI Init
# =====================================================

app = FastAPI(
    title="VoiceForge API",
    version="1.0.0",
    description="High performance TTS inference API"
)

# Thread pool for CPU-bound inference
executor = ThreadPoolExecutor(max_workers=1)


# =====================================================
# Request Schema
# =====================================================

class TTSRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Text to synthesize"
    )


# =====================================================
# Startup Event (Preload model)
# =====================================================

@app.on_event("startup")
def startup_event():
    print("[VoiceForge] Preloading model...")
    load_model()
    print("[VoiceForge] Ready for inference")


# =====================================================
# Health Check
# =====================================================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "voiceforge",
        "model": "loaded"
    }


# =====================================================
# Async wrapper for CPU inference
# =====================================================

async def synthesize_async(text: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, synthesize, text)


# =====================================================
# TTS Endpoint
# =====================================================

@app.post("/synthesize")
async def generate_audio(request: TTSRequest):

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    try:

        wav, sr = await synthesize_async(text)

        wav = np.asarray(wav).squeeze()

        # Normalize safely
        max_val = np.max(np.abs(wav))

        if max_val > 0:
            wav = wav / max_val

        wav_int16 = (wav * 32767).astype(np.int16)

        buffer = io.BytesIO()

        write(buffer, int(sr), wav_int16)

        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=voiceforge.wav",
                "X-VoiceForge": "TTS"
            }
        )

    except Exception as e:

        print(f"[VoiceForge ERROR] {str(e)}")

        raise HTTPException(
            status_code=500,
            detail="TTS generation failed"
        )
