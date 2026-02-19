from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import uuid
import os
from scipy.io.wavfile import write

from model_loader import synthesize

app = FastAPI(title="VoiceForge API")

OUTPUT_DIR = "inference"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class TTSRequest(BaseModel):
    text: str


@app.post("/synthesize")
def generate_audio(request: TTSRequest):
    wav, sr = synthesize(request.text)

    # Ensure proper types
    wav = np.array(wav).squeeze()
    sr = int(sr)

    # Avoid division by zero
    max_val = np.max(np.abs(wav))
    if max_val > 0:
        wav = wav / max_val

    # Convert to int16
    wav_int16 = (wav * 32767).astype(np.int16)

    # Unique filename
    file_id = str(uuid.uuid4())
    output_path = os.path.join(OUTPUT_DIR, f"{file_id}.wav")

    # Write WAV
    write(output_path, sr, wav_int16)

    return FileResponse(output_path, media_type="audio/wav")
