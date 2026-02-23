def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0):
    return max(min_val, min(float(value), max_val))


EMOTION_PROSODY = {
    "joy": (15, 10, 10),
    "sadness": (-20, -15, -10),
    "anger": (20, -5, 15),
    "fear": (10, 15, 5),
    "surprise": (20, 20, 10),
    "disgust": (-10, -10, 0),
    "neutral": (0, 0, 0),
}


def emotion_to_prosody(emotion: str, confidence: float):
    rate_base, pitch_base, volume_base = EMOTION_PROSODY.get(
        emotion.lower(),
        EMOTION_PROSODY["neutral"]
    )

    intensity = clamp(confidence)

    rate = int(rate_base * intensity)
    pitch = int(pitch_base * intensity)
    volume = int(volume_base * intensity)

    return rate, pitch, volume


def build_ssml(text: str, emotion: str, confidence: float):
    rate, pitch, volume = emotion_to_prosody(emotion, confidence)

    return (
        f'<speak>'
        f'<prosody rate="{rate:+d}%" pitch="{pitch:+d}%" volume="{volume:+d}%">'
        f'{text}'
        f'</prosody>'
        f'</speak>'
    )