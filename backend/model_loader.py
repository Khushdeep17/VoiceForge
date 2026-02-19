import torch
from TTS.config import load_config
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "best_model_8110.pth"))
CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "config.json"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# Load config
config = load_config(CONFIG_PATH)

# Audio processor
ap = AudioProcessor.init_from_config(config)

# Tokenizer
tokenizer, config = TTSTokenizer.init_from_config(config)

# Model
model = Vits(config, ap, tokenizer, speaker_manager=None)
model.load_checkpoint(config, MODEL_PATH)
model.to(device)
model.eval()

def synthesize(text):
    inputs = tokenizer.text_to_ids(text)
    inputs = torch.LongTensor(inputs).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model.inference(inputs)

    wav = outputs["model_outputs"][0].cpu().numpy()
    return wav, ap.sample_rate
