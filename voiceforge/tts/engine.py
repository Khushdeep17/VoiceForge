import pyttsx3

BASE_RATE = 170
BASE_VOLUME = 0.8

_EMOTION_PARAMS = {
    "joy":     (BASE_RATE + 80,  1.0, "energetic"),
    "surprise":(BASE_RATE + 60,  1.0, "excited"),
    "anger":   (BASE_RATE - 40,  0.9, "firm"),
    "sadness": (BASE_RATE - 60,  0.6, "soft"),
    "fear":    (BASE_RATE - 30,  0.7, "cautious"),
    "disgust": (BASE_RATE - 30,  0.7, "disapproving"),
    "neutral": (BASE_RATE,       BASE_VOLUME, "calm"),
}


def get_voice_params(emotion: str, intensity: float) -> tuple[int, float, str]:
    """Map emotion + intensity to (rate, volume, style).

    intensity scales the deviation from the base rate/volume.
    """
    e = emotion.lower()
    i = abs(float(intensity))

    base_rate, base_vol, style = _EMOTION_PARAMS.get(e, (BASE_RATE, BASE_VOLUME, "default"))

    # scale the delta from baseline by intensity
    rate_delta = base_rate - BASE_RATE
    vol_delta = base_vol - BASE_VOLUME

    rate = BASE_RATE + int(rate_delta * i)
    volume = BASE_VOLUME + vol_delta * i

    return rate, round(volume, 3), style


def generate_audio(text: str, rate: int, volume: float, filename: str) -> None:
    """Synthesize text to a .wav file using pyttsx3."""
    engine = pyttsx3.init()
    engine.setProperty("rate", int(rate))
    engine.setProperty("volume", float(volume))
    engine.save_to_file(text, filename)
    engine.runAndWait()
    engine.stop()
