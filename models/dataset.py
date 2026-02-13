import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class LJSpeechDataset(Dataset):
    def __init__(self, data_path="data/processed"):
        self.data_path = data_path
        self.metadata = pd.read_csv(
            os.path.join(data_path, "metadata_processed.csv")
        )

        self.mel_dir = os.path.join(data_path, "mels")
        self.wav_dir = os.path.join(data_path, "wav_norm")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        row = self.metadata.iloc[idx]
        file_id = row["id"]

        mel = np.load(
            os.path.join(self.mel_dir, f"{file_id}.npy")
        )

        wav_path = os.path.join(self.wav_dir, f"{file_id}.wav")
        wav = torch.from_numpy(
            np.load(os.path.join(self.mel_dir, f"{file_id}.npy"))
        )

        mel = torch.from_numpy(mel)

        return {
            "mel": mel,
            "id": file_id
        }

def collate_fn(batch):

    mels = [item["mel"] for item in batch]
    ids = [item["id"] for item in batch]

    # find max time dimension
    max_len = max(mel.shape[1] for mel in mels)

    padded_mels = []

    for mel in mels:
        pad_size = max_len - mel.shape[1]
        if pad_size > 0:
            pad = torch.zeros(mel.shape[0], pad_size)
            mel = torch.cat([mel, pad], dim=1)
        padded_mels.append(mel)

    padded_mels = torch.stack(padded_mels)

    return {
        "mel": padded_mels,
        "ids": ids
    }
