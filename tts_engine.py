import pyttsx3


def get_voice_params(emotion: str, intensity: float):

    base_rate = 170
    base_volume = 0.8

    if emotion == "excited":
        rate = base_rate + int(intensity * 80)
        volume = 1.0
        style = "Highly energetic and enthusiastic"

    elif emotion == "happy":
        rate = base_rate + int(intensity * 40)
        volume = 0.95
        style = "Warm and positive"

    elif emotion == "neutral":
        rate = base_rate
        volume = base_volume
        style = "Calm and balanced"

    elif emotion == "concerned":
        rate = base_rate - int(abs(intensity) * 30)
        volume = 0.7
        style = "Soft and cautious"

    elif emotion == "frustrated":
        rate = base_rate - int(abs(intensity) * 60)
        volume = 0.6
        style = "Firm and serious"

    else:
        rate = base_rate
        volume = base_volume
        style = "Default voice"

    return rate, volume, style


def generate_audio(text: str, rate: int, volume: float, filename: str):

    engine = pyttsx3.init()

    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)

    engine.save_to_file(text, filename)
    engine.runAndWait()