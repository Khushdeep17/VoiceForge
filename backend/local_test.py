from TTS.api import TTS

model_path = "models/best_model_8110.pth"
config_path = "models/config.json"

tts = TTS(model_path=model_path, config_path=config_path, gpu=False)

tts.tts_to_file(
    text="This is my custom trained voice model running locally on CPU.",
    file_path="output.wav"
)

print("Audio generated successfully.")
