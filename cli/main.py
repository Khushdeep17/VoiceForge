import os
import uuid

from voiceforge.emotion.detector import detect_emotion
from voiceforge.tts.engine import get_voice_params, generate_audio
from voiceforge.tts.personas import get_persona

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    text = input("Enter text: ").strip()
    if not text:
        print("No text provided.")
        return

    persona_name = input("Enter persona (narrator/therapist/broadcaster/assistant/storyteller) or press Enter to skip: ").strip()

    result = detect_emotion(text, mode="hybrid")
    
    persona = get_persona(persona_name) if persona_name else None
    if persona:
        rate, volume, style = persona.rate, persona.volume, persona.style
    else:
        rate, volume, style = get_voice_params(result.emotion, result.confidence)

    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)
    generate_audio(text=text, rate=rate, volume=volume, filename=filepath)

    print("\n=== VoiceForge Result ===")
    print(f"Emotion:     {result.emotion}")
    print(f"Confidence:  {round(result.confidence, 3)}")
    print(f"Voice Style: {style}  (rate={rate}, volume={volume})")
    if persona:
        print(f"Persona:     {persona_name}")
    print(f"Audio saved: {filepath}")
    if result.viz_path:
        print(f"Viz saved:   {result.viz_path}")


if __name__ == "__main__":
    main()
