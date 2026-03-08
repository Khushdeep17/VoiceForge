def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return max(min_val, min(float(value), max_val))


_EMOTION_PROSODY: dict[str, tuple[int, int, int]] = {
    "joy":      ( 15,  10,  10),
    "sadness":  (-20, -15, -10),
    "anger":    ( 20,  -5,  15),
    "fear":     ( 10,  15,   5),
    "surprise": ( 20,  20,  10),
    "disgust":  (-10, -10,   0),
    "neutral":  (  0,   0,   0),
}


def emotion_to_prosody(emotion: str, confidence: float) -> tuple[int, int, int]:
    """Return (rate%, pitch%, volume%) adjustments scaled by confidence."""
    rate_base, pitch_base, volume_base = _EMOTION_PROSODY.get(
        emotion.lower(), _EMOTION_PROSODY["neutral"]
    )
    intensity = _clamp(confidence)
    return (
        int(rate_base * intensity),
        int(pitch_base * intensity),
        int(volume_base * intensity),
    )


def build_ssml(text: str, emotion: str, confidence: float) -> str:
    """Wrap text in SSML prosody tags tuned for the given emotion."""
    rate, pitch, volume = emotion_to_prosody(emotion, confidence)
    return (
        f'<speak>'
        f'<prosody rate="{rate:+d}%" pitch="{pitch:+d}%" volume="{volume:+d}%">'
        f'{text}'
        f'</prosody>'
        f'</speak>'
    )
