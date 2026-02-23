from emotion import detect_emotion
from tts_engine import get_voice_params, generate_audio
import uuid
import os


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


text = input("Enter text: ").strip()

if not text:
    print("No text provided.")
    exit()


emotion, confidence, viz_file = detect_emotion(text, mode="hybrid")

rate, volume, style = get_voice_params(emotion, confidence)

filename = f"{uuid.uuid4()}.wav"
filepath = os.path.join(OUTPUT_DIR, filename)

generate_audio(
    text=text,
    rate=rate,
    volume=volume,
    filename=filepath
)

print("\n=== Empathy Engine Result ===")
print("Emotion:", emotion)
print("Confidence:", round(confidence, 3))
print("Voice Style:", style)
print("Audio saved:", filepath)

if viz_file:
    print("Visualization saved:", viz_file)