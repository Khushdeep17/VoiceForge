import torch
import os
from TTS.config import load_config
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer

# ---- PATHS ----
MODEL_PATH = "best_model_8110.pth"
CONFIG_PATH = "config.json"
OUTPUT_WAV = "output.wav"

# ---- LOAD CONFIG ----
config = load_config(CONFIG_PATH)

# ---- AUDIO PROCESSOR ----
ap = AudioProcessor.init_from_config(config)

# ---- TOKENIZER ----
tokenizer, config = TTSTokenizer.init_from_config(config)

# ---- MODEL ----
model = Vits(config, ap, tokenizer, speaker_manager=None)
model.load_checkpoint(config, MODEL_PATH)
model.eval()

# ---- GENERATE ----
text = "Hello bro, this is our trained voice model working locally."

wav = model.tts(text)

ap.save_wav(wav, OUTPUT_WAV)

print("✅ Audio saved as output.wav")
