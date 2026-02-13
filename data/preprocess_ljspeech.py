import os
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf

RAW_PATH = "data/raw/LJSpeech-1.1"
OUT_PATH = "data/processed"

SR = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80


def normalize_audio(audio):
    return audio / np.max(np.abs(audio))


def mel_spectrogram(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def main():

    os.makedirs(OUT_PATH, exist_ok=True)
    mel_dir = os.path.join(OUT_PATH, "mels")
    wav_dir = os.path.join(OUT_PATH, "wav_norm")

    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)

    metadata = pd.read_csv(
        os.path.join(RAW_PATH, "metadata.csv"),
        sep="|",
        header=None
    )

    metadata.columns = ["id", "text", "normalized_text"]

    print("Processing audio files...")

    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):

        wav_path = os.path.join(
            RAW_PATH,
            "wavs",
            f"{row['id']}.wav"
        )

        audio, _ = librosa.load(wav_path, sr=SR)

        audio = normalize_audio(audio)

        mel = mel_spectrogram(audio)

        np.save(os.path.join(mel_dir, f"{row['id']}.npy"), mel)
        sf.write(os.path.join(wav_dir, f"{row['id']}.wav"), audio, SR)

    metadata.to_csv(
        os.path.join(OUT_PATH, "metadata_processed.csv"),
        index=False
    )

    print("✅ Preprocessing complete.")


if __name__ == "__main__":
    main()
