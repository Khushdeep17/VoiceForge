import pyttsx3

BASE_RATE = 170
BASE_VOLUME = 0.8


def get_voice_params(emotion: str, intensity: float):

    e = emotion.lower()
    i = abs(float(intensity))

    if e == "joy":
        return BASE_RATE + int(i * 80), 1.0, "energetic"

    if e == "surprise":
        return BASE_RATE + int(i * 60), 1.0, "excited"

    if e == "anger":
        return BASE_RATE - int(i * 40), 0.9, "firm"

    if e == "sadness":
        return BASE_RATE - int(i * 60), 0.6, "soft"

    if e == "fear":
        return BASE_RATE - int(i * 30), 0.7, "cautious"

    if e == "disgust":
        return BASE_RATE - int(i * 30), 0.7, "disapproving"

    if e == "neutral":
        return BASE_RATE, BASE_VOLUME, "calm"

    return BASE_RATE, BASE_VOLUME, "default"


def generate_audio(text: str, rate: int, volume: float, filename: str):

    engine = pyttsx3.init()

    engine.setProperty("rate", int(rate))
    engine.setProperty("volume", float(volume))

    engine.save_to_file(text, filename)
    engine.runAndWait()

    engine.stop()