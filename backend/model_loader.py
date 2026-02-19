import os
import torch
from TTS.config import load_config
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer

# =====================================================
# Paths (Docker-safe absolute paths)
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(BASE_DIR, "models", "config.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model_8110.pth")

# =====================================================
# CPU Optimization
# =====================================================

torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[VoiceForge] Using device: {device}")

# =====================================================
# Global variables (lazy loading)
# =====================================================

model = None
ap = None
tokenizer = None
sample_rate = None


# =====================================================
# Load model function (called once)
# =====================================================

def load_model():
    global model, ap, tokenizer, sample_rate

    if model is not None:
        return

    print("[VoiceForge] Loading config...")
    config = load_config(CONFIG_PATH)

    print("[VoiceForge] Initializing audio processor...")
    ap = AudioProcessor.init_from_config(config)

    print("[VoiceForge] Initializing tokenizer...")
    tokenizer, config = TTSTokenizer.init_from_config(config)

    print("[VoiceForge] Loading model checkpoint...")
    model_instance = Vits(config, ap, tokenizer, speaker_manager=None)

    model_instance.load_checkpoint(config, MODEL_PATH)

    model_instance.to(device)

    model_instance.eval()

    model = model_instance
    sample_rate = ap.sample_rate

    print("[VoiceForge] Model loaded successfully.")


# =====================================================
# Synthesis function (FastAPI will call this)
# =====================================================

def synthesize(text: str):

    if model is None:
        load_model()

    inputs = tokenizer.text_to_ids(text)

    inputs = torch.LongTensor(inputs).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model.inference(inputs)

    wav = outputs["model_outputs"][0].cpu().numpy()

    return wav, sample_rate
